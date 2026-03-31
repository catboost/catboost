#include <gtest/gtest.h>
#include <mlx/mlx.h>

namespace mx = mlx::core;

// Simple helper to build a packed feature buffer for 1-byte features.
static std::vector<uint32_t> PackFeatures(const std::vector<std::vector<uint8_t>>& docs) {
    uint32_t numDocs = static_cast<uint32_t>(docs.size());
    uint32_t numFeatures = docs.empty() ? 0 : static_cast<uint32_t>(docs[0].size());
    uint32_t numUi32PerDoc = (numFeatures + 3) / 4;
    std::vector<uint32_t> packed(numDocs * numUi32PerDoc, 0);
    for (uint32_t d = 0; d < numDocs; ++d) {
        for (uint32_t f = 0; f < numFeatures; ++f) {
            uint32_t wordIdx = f / 4;
            uint32_t posInWord = f % 4;
            uint32_t shift = (3 - posInWord) * 8;
            packed[d * numUi32PerDoc + wordIdx] |= static_cast<uint32_t>(docs[d][f]) << shift;
        }
    }
    return packed;
}

TEST(MetalHistogram, PackingLayout) {
    // 4 documents, 4 features (one byte each)
    std::vector<std::vector<uint8_t>> docs = {
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 10, 11},
        {12, 13, 14, 15}
    };
    auto packed = PackFeatures(docs);

    uint32_t numDocs = 4;
    uint32_t numUi32PerDoc = 1;

    // Verify packing: doc0 should have (0 << 24) | (1 << 16) | (2 << 8) | 3
    EXPECT_EQ(packed[0], (0u << 24) | (1u << 16) | (2u << 8) | 3u);
    EXPECT_EQ(packed[1], (4u << 24) | (5u << 16) | (6u << 8) | 7u);
    EXPECT_EQ(packed[2], (8u << 24) | (9u << 16) | (10u << 8) | 11u);
    EXPECT_EQ(packed[3], (12u << 24) | (13u << 16) | (14u << 8) | 15u);

    // Transfer to MLX array and verify
    auto compressedData = mx::array(
        reinterpret_cast<const int32_t*>(packed.data()),
        {static_cast<int>(numDocs), static_cast<int>(numUi32PerDoc)},
        mx::uint32
    );
    mx::eval(compressedData);

    // Verify extraction: feature f from packed word = (word >> (24 - 8*f)) & 0xFF
    // This validates the bit layout matches what the Metal kernels expect
    for (uint32_t d = 0; d < numDocs; ++d) {
        uint32_t word = packed[d];
        for (uint32_t f = 0; f < 4; ++f) {
            uint8_t expected = docs[d][f];
            uint8_t extracted = (word >> (24 - 8 * f)) & 0xFF;
            EXPECT_EQ(extracted, expected) << "doc=" << d << " feature=" << f;
        }
    }
}

TEST(MetalHistogram, MultiWordPacking) {
    // 2 documents, 6 features -> 2 uint32 words per doc
    std::vector<std::vector<uint8_t>> docs = {
        {10, 20, 30, 40, 50, 60},
        {70, 80, 90, 100, 110, 120}
    };
    auto packed = PackFeatures(docs);

    uint32_t numUi32PerDoc = 2;  // ceil(6/4)
    EXPECT_EQ(packed.size(), 4u);  // 2 docs * 2 words

    // Doc 0, word 0: features 0-3
    EXPECT_EQ(packed[0], (10u << 24) | (20u << 16) | (30u << 8) | 40u);
    // Doc 0, word 1: features 4-5 (positions 6,7 are zero-padded)
    EXPECT_EQ(packed[1], (50u << 24) | (60u << 16) | 0u);

    // Doc 1, word 0
    EXPECT_EQ(packed[2], (70u << 24) | (80u << 16) | (90u << 8) | 100u);
    // Doc 1, word 1
    EXPECT_EQ(packed[3], (110u << 24) | (120u << 16) | 0u);
}

// TODO(Phase 7): Add Metal kernel dispatch test using mx::fast::metal_kernel()
// once the kernel dispatch infrastructure is integrated.
