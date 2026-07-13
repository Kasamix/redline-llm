#include "loader/safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <system_error>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

namespace redline {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;
using Kind = SafetensorsError::Kind;

constexpr const char* kSingleFileName = "model.safetensors";
constexpr const char* kIndexFileName = "model.safetensors.index.json";

[[noreturn]] void Fail(Kind kind, const std::string& message) {
  throw SafetensorsError(kind, "safetensors: " + message);
}

std::string ErrnoText() {
  return std::generic_category().message(errno);
}

// Little-endian u64, assembled bytewise so the read is host-endian agnostic.
std::uint64_t ReadLeU64(const unsigned char* bytes) {
  std::uint64_t value = 0;
  for (int i = 7; i >= 0; --i) {
    value = (value << 8) | static_cast<std::uint64_t>(bytes[i]);
  }
  return value;
}

// Dtype strings defined by the safetensors format that this engine can hold.
// Anything else (F8 variants, BOOL, I8/16, U16/32/64, ...) is kUnknownDtype:
// silently mis-sizing a tensor is strictly worse than refusing the file.
Dtype DtypeFromString(std::string_view s) {
  if (s == "F16")
    return Dtype::kF16;
  if (s == "BF16")
    return Dtype::kBF16;
  if (s == "F32")
    return Dtype::kF32;
  if (s == "F64")
    return Dtype::kF64;
  if (s == "I32")
    return Dtype::kI32;
  if (s == "I64")
    return Dtype::kI64;
  if (s == "U8")
    return Dtype::kU8;
  return Dtype::kUnknown;
}

// Non-negative integer extraction. nlohmann parses non-negative literals as
// number_unsigned, so this single check rejects negatives, floats, strings,
// and booleans in shape / data_offsets.
std::uint64_t RequireU64(const json& value, const std::string& tensor, const char* field,
                         const std::string& path) {
  if (!value.is_number_unsigned()) {
    Fail(Kind::kMalformedHeader, path + ": tensor '" + tensor + "': " + field +
                                     " must be a non-negative integer, got " + value.dump());
  }
  return value.get<std::uint64_t>();
}

std::string ReadWholeFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    Fail(Kind::kIo, "cannot open " + path);
  }
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

} // namespace

// --------------------------------------------------------------- SafetensorsFile

SafetensorsFile::~SafetensorsFile() {
  Release();
}

void SafetensorsFile::Release() noexcept {
  if (mapped_base_ != nullptr) {
    ::munmap(mapped_base_, mapped_size_);
    mapped_base_ = nullptr;
    mapped_size_ = 0;
  }
}

SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept
    : mapped_base_(std::exchange(other.mapped_base_, nullptr)),
      mapped_size_(std::exchange(other.mapped_size_, 0)), path_(std::move(other.path_)),
      tensors_(std::move(other.tensors_)) {}

SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& other) noexcept {
  if (this != &other) {
    Release();
    mapped_base_ = std::exchange(other.mapped_base_, nullptr);
    mapped_size_ = std::exchange(other.mapped_size_, 0);
    path_ = std::move(other.path_);
    tensors_ = std::move(other.tensors_);
  }
  return *this;
}

SafetensorsFile SafetensorsFile::Open(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    Fail(Kind::kIo, "open('" + path + "') failed: " + ErrnoText());
  }
  struct stat st = {};
  if (::fstat(fd, &st) != 0) {
    const std::string detail = ErrnoText();
    ::close(fd);
    Fail(Kind::kIo, "fstat('" + path + "') failed: " + detail);
  }
  const auto file_size = static_cast<std::uint64_t>(st.st_size);
  if (file_size < 8) {
    ::close(fd);
    Fail(Kind::kTruncatedHeader, path + ": file is " + std::to_string(file_size) +
                                     " bytes, smaller than the 8-byte header-length prefix");
  }
  void* base = ::mmap(nullptr, static_cast<std::size_t>(file_size), PROT_READ, MAP_PRIVATE, fd, 0);
  const std::string mmap_detail = (base == MAP_FAILED) ? ErrnoText() : std::string();
  ::close(fd); // the mapping keeps the file contents reachable
  if (base == MAP_FAILED) {
    Fail(Kind::kIo, "mmap('" + path + "') failed: " + mmap_detail);
  }

  SafetensorsFile file;
  file.mapped_base_ = base;
  file.mapped_size_ = static_cast<std::size_t>(file_size);
  file.path_ = path;
  file.ParseHeader(); // on throw, `file`'s destructor releases the mapping
  return file;
}

void SafetensorsFile::ParseHeader() {
  const auto* bytes = static_cast<const unsigned char*>(mapped_base_);
  const std::uint64_t header_size = ReadLeU64(bytes);
  if (header_size > mapped_size_ - 8) {
    Fail(Kind::kTruncatedHeader, path_ + ": header claims " + std::to_string(header_size) +
                                     " bytes but only " + std::to_string(mapped_size_ - 8) +
                                     " follow the length prefix");
  }
  const char* json_begin = reinterpret_cast<const char*>(bytes + 8);
  const char* json_end = json_begin + header_size;
  const std::byte* data_section = reinterpret_cast<const std::byte*>(bytes) + 8 + header_size;
  const std::uint64_t data_size = mapped_size_ - 8 - header_size;

  // Duplicate tensor names cannot be detected on the parsed DOM - a JSON
  // object silently keeps one of the colliding values - so watch key events
  // during the parse. The first key event always belongs to the root object
  // (nested objects only open after their key), and every nested key sits
  // strictly deeper, so comparing against the first key's depth isolates
  // exactly the top-level names without relying on the callback's depth base.
  std::unordered_set<std::string> seen;
  std::string duplicate;
  int root_key_depth = -1;
  json::parser_callback_t watch_keys = [&](int depth, json::parse_event_t event, json& parsed) {
    if (event == json::parse_event_t::key) {
      if (root_key_depth < 0) {
        root_key_depth = depth;
      }
      if (depth == root_key_depth && !seen.insert(parsed.get<std::string>()).second &&
          duplicate.empty()) {
        duplicate = parsed.get<std::string>();
      }
    }
    return true; // keep every value
  };

  json header;
  try {
    header = json::parse(json_begin, json_end, watch_keys, /*allow_exceptions=*/true);
  } catch (const json::exception& e) {
    Fail(Kind::kMalformedHeader, path_ + ": header is not valid JSON: " + e.what());
  }
  if (!duplicate.empty()) {
    Fail(Kind::kDuplicateName, path_ + ": duplicate tensor name '" + duplicate + "' in header");
  }
  if (!header.is_object()) {
    Fail(Kind::kMalformedHeader,
         path_ + ": header must be a JSON object, got " + std::string(header.type_name()));
  }

  for (const auto& [name, spec] : header.items()) {
    if (name == "__metadata__") {
      continue; // free-form string map written by serializers; not a tensor
    }
    if (!spec.is_object()) {
      Fail(Kind::kMalformedHeader, path_ + ": tensor '" + name + "': entry must be an object");
    }
    const auto dtype_it = spec.find("dtype");
    const auto shape_it = spec.find("shape");
    const auto offsets_it = spec.find("data_offsets");
    if (dtype_it == spec.end() || shape_it == spec.end() || offsets_it == spec.end()) {
      Fail(Kind::kMalformedHeader,
           path_ + ": tensor '" + name + "': needs dtype, shape and data_offsets");
    }

    if (!dtype_it->is_string()) {
      Fail(Kind::kMalformedHeader, path_ + ": tensor '" + name + "': dtype must be a string");
    }
    const auto dtype_str = dtype_it->get<std::string>();
    const Dtype dtype = DtypeFromString(dtype_str);
    if (dtype == Dtype::kUnknown) {
      Fail(Kind::kUnknownDtype, path_ + ": tensor '" + name + "': unknown dtype '" + dtype_str +
                                    "' (supported: F16 BF16 F32 F64 I32 I64 U8)");
    }

    if (!shape_it->is_array()) {
      Fail(Kind::kMalformedHeader, path_ + ": tensor '" + name + "': shape must be an array");
    }
    TensorView view;
    view.dtype = dtype;
    view.shape.reserve(shape_it->size());
    std::uint64_t elem_count = 1;
    for (const json& dim_json : *shape_it) {
      const std::uint64_t dim = RequireU64(dim_json, name, "shape entry", path_);
      if (dim > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
          (dim != 0 && elem_count > std::numeric_limits<std::uint64_t>::max() / dim)) {
        Fail(Kind::kMalformedHeader, path_ + ": tensor '" + name + "': shape overflows");
      }
      elem_count *= dim;
      view.shape.push_back(static_cast<std::int64_t>(dim));
    }

    if (!offsets_it->is_array() || offsets_it->size() != 2) {
      Fail(Kind::kMalformedHeader,
           path_ + ": tensor '" + name + "': data_offsets must be [begin, end]");
    }
    const std::uint64_t begin = RequireU64((*offsets_it)[0], name, "data_offsets[0]", path_);
    const std::uint64_t end = RequireU64((*offsets_it)[1], name, "data_offsets[1]", path_);
    if (begin > end || end > data_size) {
      Fail(Kind::kOffsetsOutOfBounds, path_ + ": tensor '" + name + "': data_offsets [" +
                                          std::to_string(begin) + ", " + std::to_string(end) +
                                          ") outside the " + std::to_string(data_size) +
                                          "-byte data section");
    }
    const std::uint64_t dtype_size = DtypeSize(dtype); // > 0 for every known dtype
    if (elem_count > std::numeric_limits<std::uint64_t>::max() / dtype_size) {
      Fail(Kind::kMalformedHeader, path_ + ": tensor '" + name + "': shape overflows");
    }
    const std::uint64_t expected_bytes = elem_count * dtype_size;
    if (end - begin != expected_bytes) {
      Fail(Kind::kMalformedHeader,
           path_ + ": tensor '" + name + "': data_offsets span " + std::to_string(end - begin) +
               " bytes but shape x dtype = " + std::to_string(expected_bytes) + " bytes");
    }
    view.data = data_section + begin;
    view.nbytes = static_cast<std::size_t>(end - begin);
    tensors_.emplace(name, std::move(view)); // name uniqueness established during parse
  }
}

bool SafetensorsFile::Contains(std::string_view name) const {
  return tensors_.find(name) != tensors_.end();
}

const TensorView& SafetensorsFile::Tensor(std::string_view name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    Fail(Kind::kMissingTensor, path_ + ": missing tensor: " + std::string(name));
  }
  return it->second;
}

// -------------------------------------------------------------- SafetensorsModel

SafetensorsModel SafetensorsModel::OpenDir(const std::string& model_dir) {
  std::error_code ec;
  const fs::path dir(model_dir);
  const fs::path single = dir / kSingleFileName;
  const fs::path index = dir / kIndexFileName;
  if (fs::exists(single, ec)) {
    return OpenSingle(single.string());
  }
  if (fs::exists(index, ec)) {
    return OpenIndex(index.string());
  }
  Fail(Kind::kIo, "neither " + std::string(kSingleFileName) + " nor " +
                      std::string(kIndexFileName) + " found under " + model_dir);
}

SafetensorsModel SafetensorsModel::OpenSingle(const std::string& safetensors_path) {
  SafetensorsModel model;
  model.shards_.push_back(SafetensorsFile::Open(safetensors_path));
  model.merged_ = model.shards_.front().tensors(); // one shard: plain copy of the views
  return model;
}

SafetensorsModel SafetensorsModel::OpenIndex(const std::string& index_path) {
  json index;
  try {
    index = json::parse(ReadWholeFile(index_path));
  } catch (const json::exception& e) {
    Fail(Kind::kMalformedIndex, index_path + ": not valid JSON: " + e.what());
  }
  if (!index.is_object() || !index.contains("weight_map") || !index["weight_map"].is_object()) {
    Fail(Kind::kMalformedIndex, index_path + ": expected an object with a 'weight_map' object");
  }
  const json& weight_map = index["weight_map"];

  // Open each distinct shard once, resolved relative to the index file.
  // std::map keeps shard order deterministic (by filename).
  const fs::path base_dir = fs::path(index_path).parent_path();
  std::map<std::string, std::size_t> shard_slot;
  SafetensorsModel model;
  for (const auto& [tensor_name, shard_json] : weight_map.items()) {
    if (!shard_json.is_string()) {
      Fail(Kind::kMalformedIndex,
           index_path + ": weight_map entry '" + tensor_name + "' must name a shard file (string)");
    }
    shard_slot.emplace(shard_json.get<std::string>(), 0);
  }
  for (auto& [shard_name, slot] : shard_slot) {
    slot = model.shards_.size();
    model.shards_.push_back(SafetensorsFile::Open((base_dir / shard_name).string()));
  }

  // Merge every shard's tensors into one lookup map; a name present in two
  // shard files is ambiguous no matter what weight_map claims.
  for (const SafetensorsFile& shard : model.shards_) {
    for (const auto& [name, view] : shard.tensors()) {
      if (!model.merged_.emplace(name, view).second) {
        Fail(Kind::kDuplicateName, index_path + ": tensor '" + name +
                                       "' appears in more than one shard (" + shard.path() +
                                       " and an earlier shard)");
      }
    }
  }

  // Every index entry must resolve to a tensor in exactly the shard it names.
  for (const auto& [tensor_name, shard_json] : weight_map.items()) {
    const SafetensorsFile& shard = model.shards_[shard_slot.at(shard_json.get<std::string>())];
    if (!shard.Contains(tensor_name)) {
      Fail(Kind::kMissingTensor, index_path + ": weight_map names tensor '" + tensor_name +
                                     "' in " + shard.path() + " but the shard does not contain it");
    }
  }
  return model;
}

bool SafetensorsModel::Contains(std::string_view name) const {
  return merged_.find(name) != merged_.end();
}

const TensorView& SafetensorsModel::Tensor(std::string_view name) const {
  const auto it = merged_.find(name);
  if (it == merged_.end()) {
    Fail(Kind::kMissingTensor, "model: missing tensor: " + std::string(name));
  }
  return it->second;
}

} // namespace redline
