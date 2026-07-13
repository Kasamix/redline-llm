// Python bindings for the redline engine. Module name: `redline`.
//
// Token-id-only contract: the engine consumes prompt token ids and produces
// generated token ids - no strings cross this boundary. Tokenization and
// detokenization are client-side (python/redline_client), so the serving
// path never carries text processing.
//
// The engine is compute-bound C++/CUDA, so every potentially long call -
// construction (weights load), add_request, step - runs under
// py::call_guard<py::gil_scoped_release>: a Python driver thread can overlap
// its bookkeeping with device work. Releasing the GIL enables that overlap,
// not concurrency: the engine keeps its single-driver-thread contract
// (core/engine.hpp), debug-asserted engine-side.
//
// Exception mapping (pybind11's default translators, no custom registration
// needed): std::invalid_argument from request/option validation surfaces as
// ValueError; every component error type (ConfigError, SafetensorsError,
// WeightsError, ExecutorError, GemmError) derives from std::runtime_error
// and surfaces as RuntimeError carrying the component's message (e.g. the
// itemized memory-preflight breakdown on an oversized configuration).
//
// This module is also the engine's measured path in the cross-engine
// benchmark harness (bench/FAIRNESS.md client-path rule); the separate
// redline_bench executable is a profiling tool only.

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/engine.hpp"

namespace py = pybind11;

namespace {

// StepResult.finish_reason as the documented section-10 strings.
const char* FinishReasonName(redline::FinishReason reason) {
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

redline::AdmissionPolicy ParseAdmissionPolicy(const std::string& name) {
  if (name == "reserve_full") {
    return redline::AdmissionPolicy::kReserveFull;
  }
  if (name == "watermark") {
    return redline::AdmissionPolicy::kWatermark;
  }
  throw std::invalid_argument("admission_policy must be 'reserve_full' or 'watermark', got '" +
                              name + "'");
}

// One step() emission: (req_id, token, finished, finish_reason). Built while
// the GIL is released; pybind11 converts to a list of tuples afterwards.
using StepTuple = std::tuple<redline::RequestId, redline::TokenId, bool, std::string>;

std::unique_ptr<redline::Engine> MakeEngine(const std::string& model_dir, double kv_pool_gb,
                                            std::int32_t max_batch, bool enable_cuda_graphs,
                                            std::int32_t max_seq_len, std::int32_t prefill_chunk,
                                            const std::string& admission_policy,
                                            std::int32_t pad_eager_to_bucket,
                                            const std::optional<std::string>& debug_dump_dir) {
  redline::EngineOptions options;
  options.model_dir = model_dir;
  options.kv_pool_gb = kv_pool_gb;
  options.max_batch = max_batch;
  options.enable_cuda_graphs = enable_cuda_graphs;
  options.max_seq_len = max_seq_len;
  options.prefill_chunk_tokens = prefill_chunk;
  options.admission_policy = ParseAdmissionPolicy(admission_policy);
  options.pad_eager_to_bucket = pad_eager_to_bucket;
  // None and "" both mean off (EngineOptions::debug_dump_dir empty = off).
  // When set, the executor's first prefill chunk writes its layer-0
  // activations there (docs/DESIGN.md section 12f; tests/e2e/test_layer0.py).
  options.debug_dump_dir = debug_dump_dir.value_or("");
  return std::make_unique<redline::Engine>(std::move(options));
}

} // namespace

PYBIND11_MODULE(redline, m) {
  m.doc() = "redline: single-GPU LLM inference engine (paged KV cache, continuous "
            "batching, CUDA-graph decode). The engine consumes and produces TOKEN IDS "
            "only - tokenization is client-side (see python/redline_client).";
  m.attr("__version__") = "0.1.0";

  py::class_<redline::Engine>(m, "Engine")
      .def(py::init(&MakeEngine),
           // Construction loads ~3 GiB of weights and probes cuBLASLt -
           // seconds of pure C++/CUDA work, so the GIL is released.
           py::call_guard<py::gil_scoped_release>(),
           // Defaults are the dev preset (docs/DESIGN.md sections 10-11): the
           // smallest supported GPU must be safe by default; the bench
           // harness always passes its full config explicitly.
           py::arg("model_dir"), py::arg("kv_pool_gb") = 1.0, py::arg("max_batch") = 8,
           py::arg("enable_cuda_graphs") = true, py::arg("max_seq_len") = 2048,
           py::arg("prefill_chunk") = 1024, py::arg("admission_policy") = "reserve_full",
           py::arg("pad_eager_to_bucket") = 0, py::arg("debug_dump_dir") = py::none(),
           "Load the model from `model_dir` and initialize the paged KV pool.\n\n"
           "Defaults are the dev preset. `admission_policy` is 'reserve_full' or "
           "'watermark'. Debug/eval-only kwargs (docs/DESIGN.md section 12): "
           "`pad_eager_to_bucket` (int; 0 = off, b pads every eager decode batch to "
           "the smallest configured bucket >= max(live_batch, b) with dummy rows - "
           "True == 1 means 'smallest bucket covering the live batch') and "
           "`debug_dump_dir` (None or '' = off; when set, the first prefill chunk "
           "writes its layer-0 activations there - section 12f, "
           "tests/e2e/test_layer0.py). Raises ValueError on an unknown "
           "admission_policy or an out-of-range kv_pool_gb; every other invalid "
           "option (max_batch, max_seq_len, prefill_chunk, pad_eager_to_bucket) "
           "raises RuntimeError, as does a configuration that cannot fit device "
           "memory (that message carries the itemized budget).")
      .def(
          "add_request",
          [](redline::Engine& self, std::vector<redline::TokenId> token_ids,
             std::int32_t max_new_tokens, bool ignore_eos,
             std::optional<std::vector<redline::TokenId>> forced_tokens) {
            // Each list is converted into exactly one vector by pybind11 and
            // moved into the engine - no further copies.
            return self.AddRequest(std::move(token_ids), max_new_tokens, ignore_eos,
                                   forced_tokens.has_value() ? std::move(*forced_tokens)
                                                             : std::vector<redline::TokenId>{});
          },
          py::call_guard<py::gil_scoped_release>(), py::arg("token_ids"), py::arg("max_new_tokens"),
          py::arg("ignore_eos") = false, py::arg("forced_tokens") = py::none(),
          "Queue a tokenized request; returns its request id immediately "
          "(admission happens inside step()). Raises ValueError - recording "
          "nothing - on an empty prompt, max_new_tokens <= 0, prompt_len + "
          "max_new_tokens > max_seq_len, a request that could never fit the KV "
          "pool, or any token id outside the vocabulary. ignore_eos skips the "
          "EOS stop so benchmark runs can force exact output lengths. "
          "forced_tokens (debug/eval only) is the teacher-forcing hook of "
          "docs/DESIGN.md section 12(c): forced_tokens[n] is appended as the "
          "sequence's n-th generated token while step() still reports the "
          "engine's own argmax, so parity suites can score every position "
          "along a fixed reference continuation.")
      .def(
          "step",
          [](redline::Engine& self) {
            const std::vector<redline::StepResult> results = self.Step();
            std::vector<StepTuple> out;
            out.reserve(results.size());
            for (const redline::StepResult& r : results) {
              out.emplace_back(r.request_id, r.token, r.finished,
                               FinishReasonName(r.finish_reason));
            }
            return out;
          },
          py::call_guard<py::gil_scoped_release>(),
          "Run one engine iteration - one prefill chunk or one decode batch "
          "(docs/DESIGN.md section 8) - and return every token it produced as "
          "[(req_id, token, finished, finish_reason), ...] with finish_reason "
          "in {'', 'eos', 'length', 'aborted'}. May be empty (e.g. a non-final "
          "prefill chunk, or nothing to do). Watermark-policy aborts are "
          "reported first, as (req_id, 0, True, 'aborted') - no token was "
          "produced for them. Under teacher forcing the reported token is the "
          "engine's own argmax, not the appended forced token. The GIL is "
          "released for the call.")
      .def("has_capacity", &redline::Engine::HasCapacity,
           "True while the waiting queue is below its cap (pure backpressure; "
           "KV admission happens inside step()).")
      .def(
          "stats",
          [](const redline::Engine& self) {
            const redline::EngineStats s = self.Stats();
            py::dict d;
            d["steps"] = s.steps;
            d["prefill_tokens"] = s.prefill_tokens;
            d["decode_tokens"] = s.decode_tokens;
            d["waiting"] = s.num_waiting;
            d["running"] = s.num_running;
            d["free_blocks"] = s.free_kv_blocks;
            d["reserved_blocks"] = s.reserved_kv_blocks;
            d["total_blocks"] = s.total_kv_blocks;
            d["graph_replays"] = s.graph_replays;
            d["eager_decodes"] = s.eager_decodes;
            d["aborts"] = s.aborts;
            py::dict histogram;
            for (const auto& [bucket, count] : s.bucket_histogram) {
              histogram[py::int_(bucket)] = count;
            }
            d["bucket_histogram"] = histogram;
            return d;
          },
          "Engine counters as a dict (docs/DESIGN.md section 10): total_blocks, "
          "free_blocks, reserved_blocks, waiting, running, steps, prefill_tokens, "
          "decode_tokens, graph_replays, eager_decodes, bucket_histogram "
          "({bucket: decode steps executed at exactly that bucket's shape}), "
          "aborts.")
      .def("debug_block_table", &redline::Engine::DebugBlockTable, py::arg("request_id"),
           "Debug probe (docs/DESIGN.md section 12d fragmentation audit): "
           "physical KV block ids currently owned by a live request, in "
           "allocation order. Empty for an unknown or finished id. Host-side "
           "scheduler state; zero device cost.")
      .def("algo_report", &redline::Engine::AlgoReportJson,
           "Selected-algorithm report of the engine's cuBLASLt wrapper as JSON "
           "text (docs/DESIGN.md section 12d) - recorded by bench/test output "
           "to keep shape-equality arguments auditable.");
}
