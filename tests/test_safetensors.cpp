// SafetensorsFile / SafetensorsModel unit tests (docs/DESIGN.md section 4).
//
// Every fixture file is synthesized in a per-test temporary directory at run
// time - no binary blobs live in the repository. The writer builds headers by
// plain string concatenation on purpose: a JSON library would silently
// deduplicate keys and could not produce the malformed headers (duplicate
// names, bad offsets, unknown dtypes) the typed-error tests require.

#include "loader/safetensors.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/types.hpp"

namespace redline {
namespace {

namespace fs = std::filesystem;
using Kind = SafetensorsError::Kind;

// ------------------------------------------------------------ fixture helpers

// One tensor for the synthetic writer. `bytes` is the authoritative payload;
// data_offsets are assigned sequentially in declaration order.
struct SpecTensor {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};

void AppendLeU64(std::vector<std::uint8_t>& out, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
  }
}

std::string JsonShape(const std::vector<std::int64_t>& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    out += (i ? "," : "") + std::to_string(shape[i]);
  }
  return out + "]";
}

std::string JsonEntry(const SpecTensor& t, std::uint64_t begin, std::uint64_t end) {
  return "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":" + JsonShape(t.shape) +
         ",\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
}

// [u64 LE header length][header JSON][payload] from a raw header string -
// the malformed-header tests feed deliberately broken JSON through this.
std::vector<std::uint8_t> BuildRaw(const std::string& header_json,
                                   const std::vector<std::uint8_t>& payload) {
  std::vector<std::uint8_t> file;
  AppendLeU64(file, header_json.size());
  file.insert(file.end(), header_json.begin(), header_json.end());
  file.insert(file.end(), payload.begin(), payload.end());
  return file;
}

// Well-formed file from tensor specs; offsets packed back to back in order.
// `pad_header` appends spaces to an 8-byte multiple as real serializers do.
std::vector<std::uint8_t> BuildValid(const std::vector<SpecTensor>& tensors,
                                     bool pad_header = false, bool with_metadata = false) {
  std::string header = "{";
  std::vector<std::uint8_t> payload;
  bool first = true;
  if (with_metadata) {
    header += "\"__metadata__\":{\"format\":\"pt\"}";
    first = false;
  }
  for (const SpecTensor& t : tensors) {
    const std::uint64_t begin = payload.size();
    payload.insert(payload.end(), t.bytes.begin(), t.bytes.end());
    header += (first ? "" : ",") + JsonEntry(t, begin, payload.size());
    first = false;
  }
  header += "}";
  if (pad_header) {
    while (header.size() % 8 != 0) {
      header += ' ';
    }
  }
  return BuildRaw(header, payload);
}

std::vector<std::uint8_t> PatternBytes(std::size_t n, std::uint8_t seed) {
  std::vector<std::uint8_t> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = static_cast<std::uint8_t>(seed + 7 * i);
  }
  return out;
}

void ExpectViewMatches(const TensorView& view, const SpecTensor& spec, Dtype expected_dtype) {
  EXPECT_EQ(view.dtype, expected_dtype);
  EXPECT_EQ(view.shape, spec.shape);
  ASSERT_EQ(view.nbytes, spec.bytes.size());
  if (!spec.bytes.empty()) {
    EXPECT_EQ(std::memcmp(view.data, spec.bytes.data(), spec.bytes.size()), 0)
        << spec.name << ": mapped bytes differ from written payload";
  }
}

template <typename Fn> void ExpectKind(Kind expected, Fn&& fn) {
  try {
    std::forward<Fn>(fn)();
    ADD_FAILURE() << "expected SafetensorsError, nothing was thrown";
  } catch (const SafetensorsError& e) {
    EXPECT_EQ(e.kind(), expected) << e.what();
  } catch (const std::exception& e) {
    ADD_FAILURE() << "expected SafetensorsError, got: " << e.what();
  }
}

class SafetensorsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<int> counter{0};
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ = fs::temp_directory_path() / (std::string("redline_safetensors_") + info->name() + "_" +
                                        std::to_string(counter.fetch_add(1)));
    fs::create_directories(dir_);
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove_all(dir_, ec); // best effort; never fails the test
  }

  std::string Write(const std::string& filename, const std::vector<std::uint8_t>& bytes) {
    const fs::path p = dir_ / filename;
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    EXPECT_TRUE(out.good()) << "failed writing fixture " << p;
    return p.string();
  }

  std::string WriteText(const std::string& filename, const std::string& text) {
    return Write(filename, std::vector<std::uint8_t>(text.begin(), text.end()));
  }

  fs::path dir_;
};

// ------------------------------------------------------- single-file happy path

TEST_F(SafetensorsTest, RoundTripsTwoTensorFile) {
  const SpecTensor embed{"model.embed_tokens.weight", "BF16", {4, 3}, PatternBytes(24, 0x11)};
  const SpecTensor norm{"model.norm.weight", "F32", {5}, PatternBytes(20, 0xA0)};
  const std::vector<std::uint8_t> file_bytes = BuildValid({embed, norm});
  const std::string path = Write("model.safetensors", file_bytes);

  const SafetensorsFile file = SafetensorsFile::Open(path);
  ASSERT_EQ(file.tensors().size(), 2u);
  ASSERT_TRUE(file.Contains(embed.name));
  ASSERT_TRUE(file.Contains(norm.name));
  EXPECT_FALSE(file.Contains("lm_head.weight"));

  const TensorView& embed_view = file.Tensor(embed.name);
  const TensorView& norm_view = file.Tensor(norm.name);
  ExpectViewMatches(embed_view, embed, Dtype::kBF16);
  ExpectViewMatches(norm_view, norm, Dtype::kF32);

  // Offsets: tensors were packed back to back, so the mapped views must sit
  // exactly nbytes apart, and each view must equal the raw file slice at
  // 8 + header_len + data_offsets.begin.
  EXPECT_EQ(norm_view.data - embed_view.data, static_cast<std::ptrdiff_t>(embed.bytes.size()));
  std::uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) {
    header_len = (header_len << 8) | file_bytes[static_cast<std::size_t>(i)];
  }
  const std::size_t data_base = 8 + static_cast<std::size_t>(header_len);
  EXPECT_EQ(std::memcmp(embed_view.data, file_bytes.data() + data_base, embed.bytes.size()), 0);
  EXPECT_EQ(std::memcmp(norm_view.data, file_bytes.data() + data_base + embed.bytes.size(),
                        norm.bytes.size()),
            0);
}

TEST_F(SafetensorsTest, IgnoresMetadataEntry) {
  const SpecTensor t{"w", "F16", {2, 2}, PatternBytes(8, 1)};
  const std::string path =
      Write("m.safetensors", BuildValid({t}, /*pad_header=*/false, /*with_metadata=*/true));
  const SafetensorsFile file = SafetensorsFile::Open(path);
  EXPECT_EQ(file.tensors().size(), 1u);
  EXPECT_FALSE(file.Contains("__metadata__"));
  ExpectViewMatches(file.Tensor("w"), t, Dtype::kF16);
}

TEST_F(SafetensorsTest, AcceptsSpacePaddedHeader) {
  // Serializers pad the header with 0x20 to an 8-byte boundary.
  const SpecTensor t{"w", "I64", {3}, PatternBytes(24, 9)};
  const std::string path = Write("m.safetensors", BuildValid({t}, /*pad_header=*/true));
  const SafetensorsFile file = SafetensorsFile::Open(path);
  ExpectViewMatches(file.Tensor("w"), t, Dtype::kI64);
}

TEST_F(SafetensorsTest, HandlesZeroElementTensor) {
  const SpecTensor t{"empty", "F16", {0}, {}};
  const SafetensorsFile file = SafetensorsFile::Open(Write("m.safetensors", BuildValid({t})));
  const TensorView& view = file.Tensor("empty");
  EXPECT_EQ(view.nbytes, 0u);
  EXPECT_EQ(view.shape, (std::vector<std::int64_t>{0}));
}

TEST_F(SafetensorsTest, MissingTensorLookupThrowsTyped) {
  const SpecTensor t{"w", "U8", {4}, PatternBytes(4, 3)};
  const SafetensorsFile file = SafetensorsFile::Open(Write("m.safetensors", BuildValid({t})));
  ExpectKind(Kind::kMissingTensor, [&] { (void)file.Tensor("absent"); });
  // The typed error stays catchable as std::runtime_error for generic sites.
  EXPECT_THROW((void)file.Tensor("absent"), std::runtime_error);
}

// ------------------------------------------------------------ malformed inputs

TEST_F(SafetensorsTest, FileShorterThanLengthPrefixFails) {
  const std::string path = Write("tiny.safetensors", {0x01, 0x02, 0x03});
  ExpectKind(Kind::kTruncatedHeader, [&] { (void)SafetensorsFile::Open(path); });
}

TEST_F(SafetensorsTest, HeaderLengthPastEofFails) {
  std::vector<std::uint8_t> file;
  AppendLeU64(file, 1000); // claims 1000 header bytes ...
  file.push_back('{');     // ... provides 2
  file.push_back('}');
  const std::string path = Write("truncated.safetensors", file);
  ExpectKind(Kind::kTruncatedHeader, [&] { (void)SafetensorsFile::Open(path); });
}

TEST_F(SafetensorsTest, OffsetsPastDataSectionFail) {
  // Data section holds 2 bytes; the tensor claims [0, 4).
  const std::string header = R"({"t":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}})";
  const std::string path = Write("oob.safetensors", BuildRaw(header, {0xAA, 0xBB}));
  ExpectKind(Kind::kOffsetsOutOfBounds, [&] { (void)SafetensorsFile::Open(path); });
}

TEST_F(SafetensorsTest, BeginAfterEndFails) {
  const std::string header = R"({"t":{"dtype":"F16","shape":[1],"data_offsets":[4,2]}})";
  const std::string path = Write("swap.safetensors", BuildRaw(header, PatternBytes(8, 0)));
  ExpectKind(Kind::kOffsetsOutOfBounds, [&] { (void)SafetensorsFile::Open(path); });
}

TEST_F(SafetensorsTest, UnknownDtypeFails) {
  // F8_E4M3 is valid safetensors but has no engine Dtype mapping.
  const std::string header = R"({"t":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]}})";
  const std::string path = Write("f8.safetensors", BuildRaw(header, PatternBytes(4, 0)));
  ExpectKind(Kind::kUnknownDtype, [&] { (void)SafetensorsFile::Open(path); });
}

TEST_F(SafetensorsTest, DuplicateTensorNamesFail) {
  // A JSON library would collapse the duplicate key; raw concatenation keeps it.
  const std::string header = R"({"t":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},)"
                             R"("t":{"dtype":"F16","shape":[2],"data_offsets":[4,8]}})";
  const std::string path = Write("dup.safetensors", BuildRaw(header, PatternBytes(8, 0)));
  ExpectKind(Kind::kDuplicateName, [&] { (void)SafetensorsFile::Open(path); });
}

TEST_F(SafetensorsTest, ShapeDtypeSpanMismatchFails) {
  // F16 x shape [3] = 6 bytes, but the span holds 4.
  const std::string header = R"({"t":{"dtype":"F16","shape":[3],"data_offsets":[0,4]}})";
  const std::string path = Write("span.safetensors", BuildRaw(header, PatternBytes(4, 0)));
  ExpectKind(Kind::kMalformedHeader, [&] { (void)SafetensorsFile::Open(path); });
}

TEST_F(SafetensorsTest, NonJsonHeaderFails) {
  const std::string path = Write("garbage.safetensors", BuildRaw("this is not json", {}));
  ExpectKind(Kind::kMalformedHeader, [&] { (void)SafetensorsFile::Open(path); });
}

TEST_F(SafetensorsTest, NonObjectHeaderFails) {
  const std::string path = Write("array.safetensors", BuildRaw("[1,2,3]", {}));
  ExpectKind(Kind::kMalformedHeader, [&] { (void)SafetensorsFile::Open(path); });
}

TEST_F(SafetensorsTest, MissingRequiredFieldFails) {
  const std::string header = R"({"t":{"dtype":"F16","shape":[2]}})"; // no data_offsets
  const std::string path = Write("nofield.safetensors", BuildRaw(header, PatternBytes(4, 0)));
  ExpectKind(Kind::kMalformedHeader, [&] { (void)SafetensorsFile::Open(path); });
}

TEST_F(SafetensorsTest, NegativeShapeEntryFails) {
  const std::string header = R"({"t":{"dtype":"F16","shape":[-2],"data_offsets":[0,4]}})";
  const std::string path = Write("negshape.safetensors", BuildRaw(header, PatternBytes(4, 0)));
  ExpectKind(Kind::kMalformedHeader, [&] { (void)SafetensorsFile::Open(path); });
}

TEST_F(SafetensorsTest, NonexistentFileFails) {
  ExpectKind(Kind::kIo,
             [&] { (void)SafetensorsFile::Open((dir_ / "does_not_exist.safetensors").string()); });
}

// -------------------------------------------------------------- sharded models

TEST_F(SafetensorsTest, IndexMergesTwoShards) {
  const SpecTensor a{"model.layers.0.w", "F16", {2, 1}, PatternBytes(4, 0x21)};
  const SpecTensor b{"model.layers.1.w", "F32", {1}, PatternBytes(4, 0x42)};
  const SpecTensor c{"model.norm.weight", "U8", {3}, PatternBytes(3, 0x63)};
  Write("model-00001-of-00002.safetensors", BuildValid({a}));
  Write("model-00002-of-00002.safetensors", BuildValid({b, c}));
  const std::string index_path =
      WriteText("model.safetensors.index.json",
                R"({"metadata":{"total_size":11},"weight_map":{)"
                R"("model.layers.0.w":"model-00001-of-00002.safetensors",)"
                R"("model.layers.1.w":"model-00002-of-00002.safetensors",)"
                R"("model.norm.weight":"model-00002-of-00002.safetensors"}})");

  const SafetensorsModel model = SafetensorsModel::OpenIndex(index_path);
  EXPECT_EQ(model.num_shards(), 2u);
  ASSERT_EQ(model.tensors().size(), 3u);
  ExpectViewMatches(model.Tensor(a.name), a, Dtype::kF16);
  ExpectViewMatches(model.Tensor(b.name), b, Dtype::kF32);
  ExpectViewMatches(model.Tensor(c.name), c, Dtype::kU8);
  EXPECT_FALSE(model.Contains("lm_head.weight"));
  ExpectKind(Kind::kMissingTensor, [&] { (void)model.Tensor("lm_head.weight"); });
}

TEST_F(SafetensorsTest, OpenDirPrefersSingleFile) {
  const SpecTensor t{"w", "BF16", {2}, PatternBytes(4, 5)};
  Write("model.safetensors", BuildValid({t}));
  const SafetensorsModel model = SafetensorsModel::OpenDir(dir_.string());
  EXPECT_EQ(model.num_shards(), 1u);
  ExpectViewMatches(model.Tensor("w"), t, Dtype::kBF16);
}

TEST_F(SafetensorsTest, OpenDirFallsBackToIndex) {
  const SpecTensor t{"w", "BF16", {2}, PatternBytes(4, 5)};
  Write("model-00001-of-00001.safetensors", BuildValid({t}));
  WriteText("model.safetensors.index.json",
            R"({"weight_map":{"w":"model-00001-of-00001.safetensors"}})");
  const SafetensorsModel model = SafetensorsModel::OpenDir(dir_.string());
  EXPECT_EQ(model.num_shards(), 1u);
  ExpectViewMatches(model.Tensor("w"), t, Dtype::kBF16);
}

TEST_F(SafetensorsTest, OpenDirWithoutCheckpointFails) {
  ExpectKind(Kind::kIo, [&] { (void)SafetensorsModel::OpenDir(dir_.string()); });
}

TEST_F(SafetensorsTest, IndexReferencingMissingShardFails) {
  const std::string index_path =
      WriteText("model.safetensors.index.json", R"({"weight_map":{"w":"missing.safetensors"}})");
  ExpectKind(Kind::kIo, [&] { (void)SafetensorsModel::OpenIndex(index_path); });
}

TEST_F(SafetensorsTest, IndexNamingAbsentTensorFails) {
  const SpecTensor a{"present", "F16", {1}, PatternBytes(2, 1)};
  Write("shard.safetensors", BuildValid({a}));
  const std::string index_path =
      WriteText("model.safetensors.index.json",
                R"({"weight_map":{"present":"shard.safetensors","ghost":"shard.safetensors"}})");
  ExpectKind(Kind::kMissingTensor, [&] { (void)SafetensorsModel::OpenIndex(index_path); });
}

TEST_F(SafetensorsTest, DuplicateTensorAcrossShardsFails) {
  const SpecTensor a{"dup", "F16", {1}, PatternBytes(2, 1)};
  const SpecTensor b{"other", "F16", {1}, PatternBytes(2, 2)};
  Write("shard1.safetensors", BuildValid({a}));
  Write("shard2.safetensors", BuildValid({a, b})); // "dup" appears in both shards
  const std::string index_path =
      WriteText("model.safetensors.index.json",
                R"({"weight_map":{"dup":"shard1.safetensors","other":"shard2.safetensors"}})");
  ExpectKind(Kind::kDuplicateName, [&] { (void)SafetensorsModel::OpenIndex(index_path); });
}

TEST_F(SafetensorsTest, MalformedIndexFails) {
  const std::string not_json = WriteText("model.safetensors.index.json", "nope");
  ExpectKind(Kind::kMalformedIndex, [&] { (void)SafetensorsModel::OpenIndex(not_json); });

  const std::string no_map = WriteText("no_map.index.json", R"({"metadata":{}})");
  ExpectKind(Kind::kMalformedIndex, [&] { (void)SafetensorsModel::OpenIndex(no_map); });
}

} // namespace
} // namespace redline
