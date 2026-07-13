#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"

namespace redline {

// Typed loader failure. Every malformed-input path throws this (never a bare
// std::runtime_error), so tests and the engine init sequence can distinguish
// "file is corrupt" from "wrong checkpoint" from "tensor absent". Deriving
// from std::runtime_error keeps generic catch sites working.
class SafetensorsError : public std::runtime_error {
 public:
  enum class Kind : std::uint8_t {
    kIo,                 // open/fstat/mmap failed, or an expected file is absent
    kTruncatedHeader,    // file shorter than the u64 length prefix, or the
                         // declared header length runs past end-of-file
    kMalformedHeader,    // header JSON unparseable, wrong structure, or a
                         // tensor's byte span disagrees with shape * dtype size
    kMalformedIndex,     // model.safetensors.index.json unparseable/wrong shape
    kOffsetsOutOfBounds, // data_offsets outside the data section or begin > end
    kUnknownDtype,       // dtype string with no Dtype mapping
    kDuplicateName,      // tensor name repeated in one header or across shards
    kMissingTensor,      // lookup of a name that is not present, or an index
                         // entry whose shard lacks the named tensor
  };

  SafetensorsError(Kind kind, const std::string& message)
      : std::runtime_error(message), kind_(kind) {}

  Kind kind() const noexcept { return kind_; }

 private:
  Kind kind_;
};

// Metadata plus mapped bytes for one tensor inside a .safetensors file.
struct TensorView {
  Dtype dtype = Dtype::kUnknown;
  std::vector<std::int64_t> shape;
  const std::byte* data = nullptr; // into the mmapped file; valid while the file is open
  std::size_t nbytes = 0;
};

// Transparent-hash map so Tensor()/Contains() lookups take a string_view and
// perform no allocation (acceptance requirement: allocation-free per-tensor
// lookup, including calls with string literals).
struct StringHash {
  using is_transparent = void;
  std::size_t operator()(std::string_view s) const noexcept {
    return std::hash<std::string_view>{}(s);
  }
};
using TensorMap = std::unordered_map<std::string, TensorView, StringHash, std::equal_to<>>;

// Read-only safetensors parser: JSON header via nlohmann/json (pinned
// FetchContent dependency, docs/DESIGN.md section 3), tensor bytes via POSIX
// mmap (Linux-only runtime).
//
// File layout: [u64 little-endian header_size][header_size bytes of UTF-8
// JSON][data section]. The JSON maps tensor name -> {"dtype", "shape",
// "data_offsets": [begin, end]} with offsets relative to the data section;
// a "__metadata__" entry may be present and is ignored. The file is
// memory-mapped, so TensorView::data streams straight from page cache during
// weight upload.
//
// Qwen2.5-1.5B-Instruct ships a single model.safetensors in BF16. Conversion
// to FP16 happens at upload time, not here. With tie_word_embeddings=true the
// file simply has no lm_head tensor - the caller reuses embed_tokens.
class SafetensorsFile {
 public:
  SafetensorsFile() = default;
  ~SafetensorsFile();
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  SafetensorsFile(SafetensorsFile&& other) noexcept;
  SafetensorsFile& operator=(SafetensorsFile&& other) noexcept;

  // mmap `path` read-only and parse the JSON header. Throws SafetensorsError
  // (kIo / kTruncatedHeader / kMalformedHeader / kOffsetsOutOfBounds /
  // kUnknownDtype / kDuplicateName) on I/O failure or malformed input.
  static SafetensorsFile Open(const std::string& path);

  bool Contains(std::string_view name) const;
  const TensorView& Tensor(std::string_view name) const; // throws kMissingTensor if absent
  const TensorMap& tensors() const { return tensors_; }
  const std::string& path() const { return path_; }

 private:
  void ParseHeader(); // called by Open once the mapping is in place
  void Release() noexcept;

  void* mapped_base_ = nullptr; // whole-file POSIX mmap
  std::size_t mapped_size_ = 0;
  std::string path_;
  TensorMap tensors_;
};

// A whole checkpoint: one SafetensorsFile, or several shards united by
// model.safetensors.index.json ({"metadata": {...}, "weight_map":
// {tensor name -> shard filename}}). Presents the same lookup API as a single
// file; the merged view is built once at open time so per-tensor lookup stays
// one hash probe with no allocation.
class SafetensorsModel {
 public:
  SafetensorsModel() = default;
  SafetensorsModel(const SafetensorsModel&) = delete;
  SafetensorsModel& operator=(const SafetensorsModel&) = delete;
  SafetensorsModel(SafetensorsModel&&) noexcept = default;
  SafetensorsModel& operator=(SafetensorsModel&&) noexcept = default;

  // Opens `model.safetensors` under `model_dir` if present (the
  // Qwen2.5-1.5B-Instruct layout), otherwise `model.safetensors.index.json`;
  // throws SafetensorsError{kIo} when neither exists.
  static SafetensorsModel OpenDir(const std::string& model_dir);
  // Explicit entry points (tests and tools).
  static SafetensorsModel OpenSingle(const std::string& safetensors_path);
  static SafetensorsModel OpenIndex(const std::string& index_path);

  bool Contains(std::string_view name) const;
  const TensorView& Tensor(std::string_view name) const; // throws kMissingTensor if absent
  const TensorMap& tensors() const { return merged_; }
  std::size_t num_shards() const { return shards_.size(); }

 private:
  std::vector<SafetensorsFile> shards_;
  // Values point into the shards' mappings; SafetensorsModel is move-only so
  // the mappings outlive every handed-out TensorView reference.
  TensorMap merged_;
};

} // namespace redline
