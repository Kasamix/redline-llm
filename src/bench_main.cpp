// redline_bench: pure-C++ profiling driver (nsys/NVTX runs, step-time
// histograms). It is NOT the cross-engine measured path: published
// Redline-vs-baseline numbers come exclusively from the Python harness so
// every engine shares the same client overhead (bench/FAIRNESS.md
// client-path rule). Numbers printed here never enter a comparison table.
//
// Usage:
//   redline_bench --model DIR [--kv-gb 1.0] [--max-batch 8] [--requests 64]
//                 [--prompt-len 1024] [--gen-len 256] [--graphs 1]
//                 [--ignore-eos 1] [--seed 0] [--json out.json]
//                 [--profile-after-init 1]
//   redline_bench --selftest-workload [--seed 0]
//
// Workload: the same splitmix64 token-id stream as bench/workload.py
// (normative spec in bench/FAIRNESS.md - numpy and <random> could never
// match across languages; the static_asserts below pin the identity at
// compile time and --selftest-workload proves it at run time). Metrics use
// the definitions of docs/DESIGN.md section 13: output token throughput
// (whole-window, includes prefill and says so), TTFT, ITL p50/p99,
// per-request decode rate, plus client-side Step() wall-time stats. No
// number is labeled "prefill excluded" unless computed from within-request
// windows.
//
// Flow: construct Engine -> one unmeasured warmup request (drained; excluded
// from every metric) -> optional cudaProfilerStart() (--profile-after-init,
// for nsys --capture-range=cudaProfilerApi) -> submit-while-HasCapacity +
// Step() pump loop recording a monotonic client timestamp per emitted token
// -> optional cudaProfilerStop() -> metrics + stats() dump to stdout and
// (--json) a JSON file that also carries the raw per-token timestamps. With
// --ignore-eos 1 the process exits non-zero unless every request produced
// exactly --gen-len tokens with finish_reason "length".

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include "core/engine.hpp"
#include "core/types.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::ordered_json;

// ------------------------------------------------------------------ workload

// splitmix64 mixer - bit-identical to bench/workload.py::_splitmix64
// (normative constants in bench/FAIRNESS.md, "Workload generator").
constexpr std::uint64_t SplitMix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// token_id(seed, req, pos) = 1000 + splitmix64((seed<<40)^(req<<20)^pos) % 99000
// - uniform over [1000, 100000), far below the special-token range (>=151643).
constexpr std::int32_t WorkloadTokenId(std::uint64_t seed, std::uint64_t request_id,
                                       std::uint64_t position) {
  const std::uint64_t x = (seed << 40) ^ (request_id << 20) ^ position;
  return static_cast<std::int32_t>(1000 + SplitMix64(x) % 99000);
}

// Cross-language identity pinned at compile time: the eight values below are
// bench/workload.py::selftest_lines() (doctested there) - the exact grid
// --selftest-workload prints. If either implementation drifts, this file
// stops compiling instead of silently benchmarking a different workload.
static_assert(WorkloadTokenId(0, 0, 0) == 40535);
static_assert(WorkloadTokenId(0, 0, 1) == 72465);
static_assert(WorkloadTokenId(0, 0, 2) == 39110);
static_assert(WorkloadTokenId(0, 0, 3) == 83053);
static_assert(WorkloadTokenId(0, 1, 0) == 31861);
static_assert(WorkloadTokenId(0, 1, 1) == 23711);
static_assert(WorkloadTokenId(0, 1, 2) == 79207);
static_assert(WorkloadTokenId(0, 1, 3) == 9130);

// --selftest-workload grid - mirrors workload.py SELFTEST_REQUESTS/POSITIONS.
constexpr std::int32_t kSelftestRequests[] = {0, 1};
constexpr std::int32_t kSelftestPositions[] = {0, 1, 2, 3};

std::vector<redline::TokenId> BuildPrompt(std::uint64_t seed, std::uint64_t request_index,
                                          std::int32_t prompt_len) {
  std::vector<redline::TokenId> prompt(static_cast<std::size_t>(prompt_len));
  for (std::int32_t p = 0; p < prompt_len; ++p) {
    prompt[static_cast<std::size_t>(p)] =
        WorkloadTokenId(seed, request_index, static_cast<std::uint64_t>(p));
  }
  return prompt;
}

// ----------------------------------------------------------------- utilities

[[noreturn]] void Fatal(const std::string& message) {
  std::fprintf(stderr, "[redline_bench] error: %s\n", message.c_str());
  std::exit(1);
}

const char* FinishReasonName(redline::FinishReason reason) {
  // Same mapping as the Python binding (docs/DESIGN.md section 10).
  switch (reason) {
  case redline::FinishReason::kNone:
    return "";
  case redline::FinishReason::kEos:
    return "eos";
  case redline::FinishReason::kLength:
    return "length";
  case redline::FinishReason::kAborted:
    return "aborted";
  }
  return "";
}

double SecondsSince(Clock::time_point epoch, Clock::time_point t) {
  return std::chrono::duration<double>(t - epoch).count();
}

// Linear-interpolation percentile - the same rule as
// bench/run_bench.py::percentile (numpy's default): rank = p/100 * (n-1) on
// the sorted data, so CLI numbers and harness numbers use one definition.
// Callers guarantee a non-empty input.
double Percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  if (values.size() == 1) {
    return values.front();
  }
  const double rank = (p / 100.0) * static_cast<double>(values.size() - 1);
  const std::size_t low = static_cast<std::size_t>(rank);
  const std::size_t high = std::min(low + 1, values.size() - 1);
  const double fraction = rank - static_cast<double>(low);
  return values[low] + fraction * (values[high] - values[low]);
}

// ----------------------------------------------------------------------- CLI

struct CliOptions {
  std::string model_dir;
  double kv_gb = 1.0;
  std::int32_t max_batch = 8;
  std::int64_t requests = 64;
  std::int32_t prompt_len = 1024;
  std::int32_t gen_len = 256;
  bool graphs = true;
  bool ignore_eos = true;
  std::uint64_t seed = 0;
  std::string json_path; // empty = no JSON file
  bool profile_after_init = false;
  bool selftest = false;
};

void PrintUsage(std::FILE* out) {
  std::fprintf(out, "usage: redline_bench --model DIR [options]\n"
                    "       redline_bench --selftest-workload [--seed 0]\n"
                    "\n"
                    "Pure-C++ profiling driver (nsys/NVTX). Not the cross-engine measured path:\n"
                    "published numbers come from bench/run_bench.py only (bench/FAIRNESS.md).\n"
                    "\n"
                    "options (defaults in brackets):\n"
                    "  --model DIR              model directory: config.json + *.safetensors\n"
                    "  --kv-gb F                paged KV pool budget in GiB [1.0]\n"
                    "  --max-batch N            decode batch cap / largest graph bucket [8]\n"
                    "  --requests N             workload requests, all submitted at t=0 [64]\n"
                    "  --prompt-len N           prompt tokens per request [1024]\n"
                    "  --gen-len N              new tokens per request [256]\n"
                    "  --graphs 0|1             capture decode CUDA graphs at init [1]\n"
                    "  --ignore-eos 0|1         force exactly gen-len tokens per request; the\n"
                    "                           process exits non-zero on any shortfall [1]\n"
                    "  --seed N                 workload seed (splitmix64 stream) [0]\n"
                    "  --json PATH              write metrics + stats() dump + per-token\n"
                    "                           timestamps as JSON\n"
                    "  --profile-after-init 0|1 cudaProfilerStart() after init+warmup and\n"
                    "                           cudaProfilerStop() at end, for\n"
                    "                           nsys --capture-range=cudaProfilerApi [0]\n"
                    "  --selftest-workload      print the 8 normative token IDs (req in {0,1},\n"
                    "                           pos in {0..3}) byte-identically to\n"
                    "                           `python -m bench.workload --selftest`, then exit\n"
                    "  -h, --help               this text\n");
}

[[noreturn]] void UsageError(const std::string& message) {
  std::fprintf(stderr, "redline_bench: %s\n", message.c_str());
  PrintUsage(stderr);
  std::exit(2);
}

std::int64_t ParseInt(const char* flag, const char* text, std::int64_t lo, std::int64_t hi) {
  errno = 0;
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    UsageError(std::string(flag) + " expects an integer, got '" + text + "'");
  }
  if (value < lo || value > hi) {
    UsageError(std::string(flag) + " out of range [" + std::to_string(lo) + ", " +
               std::to_string(hi) + "]: " + text);
  }
  return value;
}

std::uint64_t ParseUint64(const char* flag, const char* text) {
  errno = 0;
  char* end = nullptr;
  if (text[0] == '-') {
    UsageError(std::string(flag) + " expects a non-negative integer, got '" + text + "'");
  }
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    UsageError(std::string(flag) + " expects a non-negative integer, got '" + text + "'");
  }
  return value;
}

double ParseDouble(const char* flag, const char* text) {
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value)) {
    UsageError(std::string(flag) + " expects a finite number, got '" + text + "'");
  }
  return value;
}

bool ParseBool01(const char* flag, const char* text) {
  if (std::strcmp(text, "0") == 0) {
    return false;
  }
  if (std::strcmp(text, "1") == 0) {
    return true;
  }
  UsageError(std::string(flag) + " expects 0 or 1, got '" + text + "'");
}

CliOptions ParseCli(int argc, char** argv) {
  CliOptions opt;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto value = [&]() -> const char* {
      if (i + 1 >= argc) {
        UsageError(flag + " expects a value");
      }
      return argv[++i];
    };
    if (flag == "--model") {
      opt.model_dir = value();
    } else if (flag == "--kv-gb") {
      opt.kv_gb = ParseDouble("--kv-gb", value());
      if (opt.kv_gb <= 0.0) {
        UsageError("--kv-gb must be > 0");
      }
    } else if (flag == "--max-batch") {
      opt.max_batch = static_cast<std::int32_t>(ParseInt("--max-batch", value(), 1, 65536));
    } else if (flag == "--requests") {
      opt.requests = ParseInt("--requests", value(), 1, 1000000);
    } else if (flag == "--prompt-len") {
      opt.prompt_len = static_cast<std::int32_t>(ParseInt("--prompt-len", value(), 1, 100000000));
    } else if (flag == "--gen-len") {
      opt.gen_len = static_cast<std::int32_t>(ParseInt("--gen-len", value(), 1, 100000000));
    } else if (flag == "--graphs") {
      opt.graphs = ParseBool01("--graphs", value());
    } else if (flag == "--ignore-eos") {
      opt.ignore_eos = ParseBool01("--ignore-eos", value());
    } else if (flag == "--seed") {
      opt.seed = ParseUint64("--seed", value());
    } else if (flag == "--json") {
      opt.json_path = value();
    } else if (flag == "--profile-after-init") {
      opt.profile_after_init = ParseBool01("--profile-after-init", value());
    } else if (flag == "--selftest-workload") {
      opt.selftest = true;
    } else if (flag == "-h" || flag == "--help") {
      PrintUsage(stdout);
      std::exit(0);
    } else {
      UsageError("unknown flag: " + flag);
    }
  }
  if (!opt.selftest && opt.model_dir.empty()) {
    UsageError("--model is required");
  }
  return opt;
}

// ------------------------------------------------------------------ selftest

int RunSelftest(std::uint64_t seed) {
  // Byte-identical to `python -m bench.workload --selftest [--seed N]`
  // (workload.py::selftest_lines): one line per (req, pos), req-major.
  for (const std::int32_t req : kSelftestRequests) {
    for (const std::int32_t pos : kSelftestPositions) {
      std::printf(
          "seed=%llu req=%d pos=%d id=%d\n", static_cast<unsigned long long>(seed), req, pos,
          WorkloadTokenId(seed, static_cast<std::uint64_t>(req), static_cast<std::uint64_t>(pos)));
    }
  }
  return 0;
}

// --------------------------------------------------------------- run + report

struct RequestTrace {
  redline::RequestId engine_id = 0;
  std::int64_t workload_index = -1;
  double submit_s = 0.0;             // relative to the measured-phase epoch
  std::vector<double> token_times_s; // one client timestamp per emitted token
  std::vector<redline::TokenId> token_ids;
  bool finished = false;
  const char* finish_reason = "";
};

struct Summary {
  double wall_window_s = 0.0;
  std::int64_t total_new_tokens = 0;
  std::int64_t requests_with_tokens = 0;
  double output_tok_s = 0.0;
  double ttft_p50_ms = 0.0;
  double ttft_p99_ms = 0.0;
  bool has_itl = false;
  double itl_p50_ms = 0.0;
  double itl_p99_ms = 0.0;
  bool has_decode_rate = false;
  double decode_rate_p50_tok_s = 0.0;
  double decode_rate_min_tok_s = 0.0;
  double decode_rate_max_tok_s = 0.0;
  double step_p50_ms = 0.0;
  double step_p99_ms = 0.0;
  double step_mean_ms = 0.0;
  double step_max_ms = 0.0;
};

// Section-13 metric formulas over the recorded traces (bench/FAIRNESS.md
// "Metrics"; mirrors bench/run_bench.py::compute_metrics conventions -
// single-token requests contribute no ITL/decode rate, zero-span requests
// contribute no decode rate).
Summary Summarize(const std::vector<RequestTrace>& traces, const std::vector<double>& step_ms) {
  Summary s;
  std::vector<double> ttft_ms;
  std::vector<double> itl_ms;
  std::vector<double> rates;
  double min_submit = std::numeric_limits<double>::infinity();
  double max_last = -std::numeric_limits<double>::infinity();
  for (const RequestTrace& tr : traces) {
    min_submit = std::min(min_submit, tr.submit_s);
    const std::vector<double>& t = tr.token_times_s;
    s.total_new_tokens += static_cast<std::int64_t>(t.size());
    if (t.empty()) {
      continue; // aborted before its first token; counted via finish reasons
    }
    ++s.requests_with_tokens;
    max_last = std::max(max_last, t.back());
    ttft_ms.push_back((t.front() - tr.submit_s) * 1e3);
    for (std::size_t i = 1; i < t.size(); ++i) {
      itl_ms.push_back((t[i] - t[i - 1]) * 1e3);
    }
    if (t.size() >= 2 && t.back() > t.front()) {
      rates.push_back(static_cast<double>(t.size() - 1) / (t.back() - t.front()));
    }
  }
  if (s.requests_with_tokens == 0) {
    Fatal("no request produced any token");
  }
  s.wall_window_s = max_last - min_submit;
  if (!(s.wall_window_s > 0.0)) {
    Fatal("non-positive wall window; timestamps are inconsistent");
  }
  s.output_tok_s = static_cast<double>(s.total_new_tokens) / s.wall_window_s;
  s.ttft_p50_ms = Percentile(ttft_ms, 50.0);
  s.ttft_p99_ms = Percentile(ttft_ms, 99.0);
  if (!itl_ms.empty()) {
    s.has_itl = true;
    s.itl_p50_ms = Percentile(itl_ms, 50.0);
    s.itl_p99_ms = Percentile(itl_ms, 99.0);
  }
  if (!rates.empty()) {
    s.has_decode_rate = true;
    s.decode_rate_p50_tok_s = Percentile(rates, 50.0);
    const auto [mn, mx] = std::minmax_element(rates.begin(), rates.end());
    s.decode_rate_min_tok_s = *mn;
    s.decode_rate_max_tok_s = *mx;
  }
  double sum = 0.0;
  double mx = 0.0;
  for (const double v : step_ms) {
    sum += v;
    mx = std::max(mx, v);
  }
  s.step_p50_ms = Percentile(step_ms, 50.0);
  s.step_p99_ms = Percentile(step_ms, 99.0);
  s.step_mean_ms = sum / static_cast<double>(step_ms.size());
  s.step_max_ms = mx;
  return s;
}

Json StatsJson(const redline::EngineStats& s) {
  // Key set and order mirror redline.Engine.stats() (docs/DESIGN.md
  // section 10). Counters are cumulative since engine init, so the warmup
  // request's prefill/decode work is included.
  Json j;
  j["steps"] = s.steps;
  j["prefill_tokens"] = s.prefill_tokens;
  j["decode_tokens"] = s.decode_tokens;
  j["waiting"] = s.num_waiting;
  j["running"] = s.num_running;
  j["free_blocks"] = s.free_kv_blocks;
  j["reserved_blocks"] = s.reserved_kv_blocks;
  j["total_blocks"] = s.total_kv_blocks;
  j["graph_replays"] = s.graph_replays;
  j["eager_decodes"] = s.eager_decodes;
  j["aborts"] = s.aborts;
  Json histogram = Json::object();
  for (const auto& [bucket, count] : s.bucket_histogram) {
    histogram[std::to_string(bucket)] = count;
  }
  j["bucket_histogram"] = histogram;
  return j;
}

int Run(const CliOptions& opt) {
  redline::EngineOptions eng; // dev-preset defaults (core/engine.hpp)
  eng.model_dir = opt.model_dir;
  eng.kv_pool_gb = opt.kv_gb;
  eng.max_batch = opt.max_batch;
  eng.enable_cuda_graphs = opt.graphs;
  // The flag set is fixed, so the per-request sequence cap is derived from
  // the workload instead of exposed as a flag: prompt_len + gen_len must fit
  // max_seq_len or every AddRequest would be rejected (core/engine.hpp).
  // Everything else (prefill_chunk_tokens, admission policy) stays at the
  // dev-preset default.
  const std::int32_t needed_seq_len = opt.prompt_len + opt.gen_len;
  const bool seq_len_derived = needed_seq_len > eng.max_seq_len;
  if (seq_len_derived) {
    eng.max_seq_len = needed_seq_len;
  }

  std::printf("[redline_bench] profiling driver; published cross-engine numbers come from\n"
              "[redline_bench] bench/run_bench.py only (bench/FAIRNESS.md client-path rule)\n");
  std::printf("[redline_bench] model: %s\n", opt.model_dir.c_str());
  std::printf("[redline_bench] engine: kv_pool_gb=%.3f max_batch=%d graphs=%d max_seq_len=%d%s "
              "prefill_chunk=%d admission=%s\n",
              eng.kv_pool_gb, eng.max_batch, eng.enable_cuda_graphs ? 1 : 0, eng.max_seq_len,
              seq_len_derived ? " (derived: prompt_len + gen_len)" : "", eng.prefill_chunk_tokens,
              eng.admission_policy == redline::AdmissionPolicy::kReserveFull ? "reserve_full"
                                                                             : "watermark");
  std::printf("[redline_bench] workload: requests=%lld prompt_len=%d gen_len=%d seed=%llu "
              "ignore_eos=%d (splitmix64, bench/FAIRNESS.md)\n",
              static_cast<long long>(opt.requests), opt.prompt_len, opt.gen_len,
              static_cast<unsigned long long>(opt.seed), opt.ignore_eos ? 1 : 0);

  redline::Engine engine(eng); // init logs its own budget/load lines on stderr

  // Unmeasured warmup: one request drained to completion so first-use costs
  // (module loads, allocator/scheduler cold paths) sit outside both the
  // metric window and the optional profiler capture range. Its token ids
  // come from workload index `requests` - past the measured grid - and it
  // always runs with ignore_eos for a deterministic length.
  const std::int32_t warmup_gen = std::min<std::int32_t>(8, opt.gen_len);
  const std::int32_t warmup_prompt =
      std::min<std::int32_t>(opt.prompt_len, eng.max_seq_len - warmup_gen);
  std::int64_t warmup_tokens = 0;
  {
    const redline::RequestId warmup_id = engine.AddRequest(
        BuildPrompt(opt.seed, static_cast<std::uint64_t>(opt.requests), warmup_prompt), warmup_gen,
        /*ignore_eos=*/true);
    bool warmup_done = false;
    while (!warmup_done) {
      const std::vector<redline::StepResult> results = engine.Step();
      for (const redline::StepResult& res : results) {
        if (res.request_id != warmup_id) {
          Fatal("warmup: emission for an unknown request id");
        }
        if (res.finish_reason != redline::FinishReason::kAborted) {
          ++warmup_tokens;
        }
        if (res.finished) {
          warmup_done = true;
        }
      }
      if (results.empty()) {
        const redline::EngineStats s = engine.Stats();
        if (s.num_waiting + s.num_running == 0) {
          Fatal("warmup: engine idle before the warmup request finished");
        }
      }
    }
    if (warmup_tokens != warmup_gen) {
      std::fprintf(stderr, "[redline_bench] warning: warmup produced %lld tokens, expected %d\n",
                   static_cast<long long>(warmup_tokens), warmup_gen);
    }
  }
  std::printf("[redline_bench] warmup: 1 request (prompt_len=%d gen_len=%d), excluded from "
              "metrics\n",
              warmup_prompt, warmup_gen);

  if (opt.profile_after_init) {
    // nsys --capture-range=cudaProfilerApi opens its window here - after
    // init + warmup - so the capture holds only steady-state stepping
    // (docs/DESIGN.md section 14 no-alloc audit).
    const cudaError_t err = cudaProfilerStart();
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[redline_bench] warning: cudaProfilerStart: %s\n",
                   cudaGetErrorString(err));
    }
  }

  // Measured pump loop: submit while HasCapacity (the full closed wave for
  // any request count below the 1024 waiting cap), then Step until every
  // request finished. Every emitted token gets the post-Step timestamp of
  // its iteration; tokens emitted by one Step share one timestamp.
  std::vector<RequestTrace> traces(static_cast<std::size_t>(opt.requests));
  std::unordered_map<redline::RequestId, std::size_t> index_by_id;
  index_by_id.reserve(traces.size());
  std::vector<double> step_ms;
  const Clock::time_point epoch = Clock::now();
  std::int64_t submitted = 0;
  std::int64_t finished = 0;
  while (finished < opt.requests) {
    while (submitted < opt.requests && engine.HasCapacity()) {
      RequestTrace& tr = traces[static_cast<std::size_t>(submitted)];
      tr.workload_index = submitted;
      std::vector<redline::TokenId> prompt =
          BuildPrompt(opt.seed, static_cast<std::uint64_t>(submitted), opt.prompt_len);
      tr.submit_s = SecondsSince(epoch, Clock::now());
      tr.engine_id = engine.AddRequest(std::move(prompt), opt.gen_len, opt.ignore_eos);
      index_by_id.emplace(tr.engine_id, static_cast<std::size_t>(submitted));
      ++submitted;
    }
    const Clock::time_point t0 = Clock::now();
    const std::vector<redline::StepResult> results = engine.Step();
    const Clock::time_point t1 = Clock::now();
    step_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    const double token_time_s = SecondsSince(epoch, t1);
    for (const redline::StepResult& res : results) {
      const auto it = index_by_id.find(res.request_id);
      if (it == index_by_id.end()) {
        Fatal("Step() emitted an unknown request id");
      }
      RequestTrace& tr = traces[it->second];
      if (res.finish_reason != redline::FinishReason::kAborted) {
        // A watermark abort is reported as token 0 with finished=true and no
        // token produced (core/types.hpp); everything else is a real token.
        tr.token_times_s.push_back(token_time_s);
        tr.token_ids.push_back(res.token);
      }
      if (res.finished) {
        tr.finished = true;
        tr.finish_reason = FinishReasonName(res.finish_reason);
        ++finished;
      }
    }
    if (results.empty() && finished < opt.requests) {
      // Non-final prefill chunks legitimately emit nothing, but then the
      // engine still reports waiting/running work. An idle engine with
      // unfinished requests means an emission was lost - stop instead of
      // spinning forever.
      const redline::EngineStats s = engine.Stats();
      if (s.num_waiting + s.num_running == 0) {
        Fatal("engine idle with unfinished requests (lost emission)");
      }
    }
  }

  if (opt.profile_after_init) {
    const cudaError_t err = cudaProfilerStop();
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[redline_bench] warning: cudaProfilerStop: %s\n",
                   cudaGetErrorString(err));
    }
  }

  const Summary sum = Summarize(traces, step_ms);
  const redline::EngineStats stats = engine.Stats();

  std::map<std::string, std::int64_t> finish_reasons;
  for (const RequestTrace& tr : traces) {
    ++finish_reasons[tr.finish_reason];
  }

  // ------------------------------------------------------------------- JSON
  Json j;
  j["tool"] = "redline_bench";
  j["role"] = "profiling driver; cross-engine numbers come from bench/run_bench.py only "
              "(bench/FAIRNESS.md client-path rule)";
  {
    Json cfg;
    cfg["model_dir"] = opt.model_dir;
    cfg["kv_pool_gb"] = eng.kv_pool_gb;
    cfg["max_batch"] = eng.max_batch;
    cfg["enable_cuda_graphs"] = eng.enable_cuda_graphs;
    cfg["max_seq_len"] = eng.max_seq_len;
    cfg["max_seq_len_derived_from_workload"] = seq_len_derived;
    cfg["prefill_chunk_tokens"] = eng.prefill_chunk_tokens;
    cfg["admission_policy"] = eng.admission_policy == redline::AdmissionPolicy::kReserveFull
                                  ? "reserve_full"
                                  : "watermark";
    cfg["profile_after_init"] = opt.profile_after_init;
    j["config"] = cfg;
  }
  {
    Json wl;
    wl["generator"] = "splitmix64 (bench/FAIRNESS.md; bit-identical to bench/workload.py)";
    wl["seed"] = opt.seed;
    wl["requests"] = opt.requests;
    wl["prompt_len"] = opt.prompt_len;
    wl["gen_len"] = opt.gen_len;
    wl["ignore_eos"] = opt.ignore_eos;
    j["workload"] = wl;
  }
  {
    Json wu;
    wu["prompt_len"] = warmup_prompt;
    wu["gen_len"] = warmup_gen;
    wu["tokens_produced"] = warmup_tokens;
    wu["note"] = "one unmeasured warmup request; excluded from metrics, included in stats";
    j["warmup"] = wu;
  }
  j["output_tok_s"] = sum.output_tok_s;
  j["output_tok_s_definition"] =
      "sum_i n_i / (max_i t_i,last - min_i t_i,submit) - total new tokens over the full wall "
      "window, which includes all prefill work (docs/DESIGN.md section 13)";
  j["total_new_tokens"] = sum.total_new_tokens;
  j["wall_window_s"] = sum.wall_window_s;
  j["num_requests"] = opt.requests;
  j["requests_with_tokens"] = sum.requests_with_tokens;
  j["ttft_p50_ms"] = sum.ttft_p50_ms;
  j["ttft_p99_ms"] = sum.ttft_p99_ms;
  if (sum.has_itl) {
    j["itl_p50_ms"] = sum.itl_p50_ms;
    j["itl_p99_ms"] = sum.itl_p99_ms;
  }
  if (sum.has_decode_rate) {
    j["decode_rate_p50_tok_s"] = sum.decode_rate_p50_tok_s;
    j["decode_rate_min_tok_s"] = sum.decode_rate_min_tok_s;
    j["decode_rate_max_tok_s"] = sum.decode_rate_max_tok_s;
  }
  j["step_time_p50_ms"] = sum.step_p50_ms;
  j["step_time_p99_ms"] = sum.step_p99_ms;
  j["step_time_mean_ms"] = sum.step_mean_ms;
  j["step_time_max_ms"] = sum.step_max_ms;
  j["client_step_calls"] = static_cast<std::int64_t>(step_ms.size());
  j["step_times_ms"] = step_ms;
  {
    Json fr = Json::object();
    for (const auto& [reason, count] : finish_reasons) {
      fr[reason] = count;
    }
    j["finish_reasons"] = fr;
  }
  j["stats"] = StatsJson(stats);
  try {
    j["algo_report"] = Json::parse(engine.AlgoReportJson());
  } catch (const std::exception&) {
    j["algo_report"] = engine.AlgoReportJson(); // raw text if not valid JSON
  }
  {
    Json arr = Json::array();
    for (const RequestTrace& tr : traces) {
      Json r;
      r["request_id"] = tr.engine_id;
      r["workload_index"] = tr.workload_index;
      r["submit_s"] = tr.submit_s;
      r["num_tokens"] = static_cast<std::int64_t>(tr.token_times_s.size());
      r["finish_reason"] = tr.finish_reason;
      if (!tr.token_times_s.empty()) {
        r["first_token_s"] = tr.token_times_s.front();
        r["last_token_s"] = tr.token_times_s.back();
      }
      r["token_times_s"] = tr.token_times_s;
      r["token_ids"] = tr.token_ids;
      arr.push_back(std::move(r));
    }
    j["per_request"] = arr;
  }
  if (!opt.json_path.empty()) {
    std::ofstream out(opt.json_path, std::ios::binary);
    if (!out) {
      Fatal("cannot open --json path for writing: " + opt.json_path);
    }
    out << j.dump(2) << '\n';
    out.flush();
    if (!out) {
      Fatal("failed writing --json file: " + opt.json_path);
    }
  }

  // ----------------------------------------------------------------- stdout
  std::printf("[redline_bench] results (docs/DESIGN.md section 13 definitions):\n");
  std::printf("  output token throughput: %.2f tok/s over the full wall window "
              "(includes all prefill work)\n",
              sum.output_tok_s);
  std::printf("  wall window: %.3f s   new tokens: %lld   requests: %lld (with >=1 token: "
              "%lld)\n",
              sum.wall_window_s, static_cast<long long>(sum.total_new_tokens),
              static_cast<long long>(opt.requests),
              static_cast<long long>(sum.requests_with_tokens));
  std::printf("  TTFT ms: p50 %.2f  p99 %.2f\n", sum.ttft_p50_ms, sum.ttft_p99_ms);
  if (sum.has_itl) {
    std::printf("  ITL ms (pooled, first token excluded): p50 %.3f  p99 %.3f\n", sum.itl_p50_ms,
                sum.itl_p99_ms);
  } else {
    std::printf("  ITL: n/a (needs >= 2 tokens per request)\n");
  }
  if (sum.has_decode_rate) {
    std::printf("  per-request decode rate tok/s: p50 %.2f  min %.2f  max %.2f\n",
                sum.decode_rate_p50_tok_s, sum.decode_rate_min_tok_s, sum.decode_rate_max_tok_s);
  }
  std::printf("  step time ms over %lld Step() calls: p50 %.3f  p99 %.3f  mean %.3f  max "
              "%.3f\n",
              static_cast<long long>(step_ms.size()), sum.step_p50_ms, sum.step_p99_ms,
              sum.step_mean_ms, sum.step_max_ms);
  std::printf("  finish reasons:");
  for (const auto& [reason, count] : finish_reasons) {
    std::printf(" %s=%lld", reason.empty() ? "(none)" : reason.c_str(),
                static_cast<long long>(count));
  }
  std::printf("\n");
  std::printf("  engine stats: steps=%llu prefill_tokens=%llu decode_tokens=%llu "
              "graph_replays=%llu eager_decodes=%llu aborts=%llu\n",
              static_cast<unsigned long long>(stats.steps),
              static_cast<unsigned long long>(stats.prefill_tokens),
              static_cast<unsigned long long>(stats.decode_tokens),
              static_cast<unsigned long long>(stats.graph_replays),
              static_cast<unsigned long long>(stats.eager_decodes),
              static_cast<unsigned long long>(stats.aborts));
  std::printf("  kv blocks: free=%lld reserved=%lld total=%lld\n",
              static_cast<long long>(stats.free_kv_blocks),
              static_cast<long long>(stats.reserved_kv_blocks),
              static_cast<long long>(stats.total_kv_blocks));
  std::printf("  bucket histogram:");
  for (const auto& [bucket, count] : stats.bucket_histogram) {
    std::printf(" %d:%llu", bucket, static_cast<unsigned long long>(count));
  }
  std::printf("\n");
  if (!opt.json_path.empty()) {
    std::printf("  json: %s\n", opt.json_path.c_str());
  }

  // Forced-length gate (bench/FAIRNESS.md "Cases"): with
  // --ignore-eos 1 every request must have produced exactly gen_len tokens
  // and stopped on the length condition. Checked after the JSON is written
  // so a violating run still leaves its artifacts for inspection.
  if (opt.ignore_eos) {
    std::int64_t violations = 0;
    for (const RequestTrace& tr : traces) {
      const bool ok = tr.finished && std::strcmp(tr.finish_reason, "length") == 0 &&
                      static_cast<std::int64_t>(tr.token_times_s.size()) == opt.gen_len;
      if (!ok) {
        ++violations;
        std::fprintf(stderr,
                     "[redline_bench] forced-length violation: request %lld (id %llu) produced "
                     "%zu tokens, expected %d (finish_reason='%s')\n",
                     static_cast<long long>(tr.workload_index),
                     static_cast<unsigned long long>(tr.engine_id), tr.token_times_s.size(),
                     opt.gen_len, tr.finish_reason);
      }
    }
    if (violations > 0) {
      std::fprintf(stderr, "[redline_bench] %lld forced-length violation(s) with --ignore-eos 1\n",
                   static_cast<long long>(violations));
      return 1;
    }
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  const CliOptions opt = ParseCli(argc, argv);
  if (opt.selftest) {
    return RunSelftest(opt.seed);
  }
  try {
    return Run(opt);
  } catch (const std::exception& e) {
    // Engine failures (config mismatch, the itemized memory-preflight
    // breakdown, executor errors) carry multi-line messages - print verbatim.
    std::fprintf(stderr, "[redline_bench] fatal: %s\n", e.what());
    return 1;
  }
}
