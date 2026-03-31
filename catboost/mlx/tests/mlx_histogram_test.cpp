#include <gtest/gtest.h>
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <catboost/mlx/kernels/kernel_sources.h>

#include <cstdint>
#include <vector>
#include <numeric>

namespace mx = mlx::core;

// Helper to build a packed feature buffer for 1-byte features.
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

// =============================================================================
// Metal kernel dispatch integration test
//
// Creates a small synthetic dataset, dispatches the histogram kernel on GPU,
// and compares the output against a CPU reference computation.
// =============================================================================

TEST(MetalHistogram, KernelDispatchSinglePartition) {
    // Setup: 8 documents, 4 features, all in one partition (depth=0).
    //
    // Feature bins:
    //   doc0: [0, 1, 0, 2]
    //   doc1: [1, 0, 1, 0]
    //   doc2: [2, 1, 2, 1]
    //   doc3: [0, 2, 0, 2]
    //   doc4: [1, 1, 1, 1]
    //   doc5: [2, 0, 2, 0]
    //   doc6: [0, 0, 0, 0]
    //   doc7: [2, 2, 2, 2]
    //
    // Gradients: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    //
    // Each feature has 3 bins (0, 1, 2), so folds=2 (split candidates between bins).
    // totalBinFeatures = 4 features * 2 folds = 8

    const uint32_t numDocs = 8;
    const uint32_t numFeatures = 4;
    const uint32_t numUi32PerDoc = 1;  // 4 features fit in 1 uint32
    const uint32_t folds = 2;          // bins 0,1,2 -> 2 folds (split after bin 0, after bin 1)
    const uint32_t totalBinFeatures = numFeatures * folds;  // 8
    const uint32_t numPartitions = 1;
    const uint32_t numStats = 1;  // gradient only

    std::vector<std::vector<uint8_t>> docs = {
        {0, 1, 0, 2},
        {1, 0, 1, 0},
        {2, 1, 2, 1},
        {0, 2, 0, 2},
        {1, 1, 1, 1},
        {2, 0, 2, 0},
        {0, 0, 0, 0},
        {2, 2, 2, 2}
    };
    auto packed = PackFeatures(docs);
    std::vector<float> gradients = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    // --- CPU reference computation ---
    // The kernel writes histogram[firstFold + bin] = sum of gradient for docs where
    // feature_bin == bin+1 (bin 0 is "below first border", kernel reads sharedHist[f*256+bin+1])
    //
    // For feature f, the kernel accumulates into sharedHist[f*256 + bin] for each doc,
    // then writes sharedHist[f*256 + bin+1] to histogram[firstFold + bin] for bin in 0..folds-1.
    // So: histogram[firstFold + 0] = sum of grads where feature_bin == 1
    //     histogram[firstFold + 1] = sum of grads where feature_bin == 2
    //
    // Feature 0 (bins: 0,1,2,0,1,2,0,2):
    //   bin=1: docs 1,4 -> 2.0+5.0 = 7.0
    //   bin=2: docs 2,5,7 -> 3.0+6.0+8.0 = 17.0
    //
    // Feature 1 (bins: 1,0,1,2,1,0,0,2):
    //   bin=1: docs 0,2,4 -> 1.0+3.0+5.0 = 9.0
    //   bin=2: docs 3,7 -> 4.0+8.0 = 12.0
    //
    // Feature 2 (bins: 0,1,2,0,1,2,0,2):
    //   bin=1: docs 1,4 -> 2.0+5.0 = 7.0
    //   bin=2: docs 2,5,7 -> 3.0+6.0+8.0 = 17.0
    //
    // Feature 3 (bins: 2,0,1,2,1,0,0,2):
    //   bin=1: docs 2,4 -> 3.0+5.0 = 8.0
    //   bin=2: docs 0,3,7 -> 1.0+4.0+8.0 = 13.0

    std::vector<float> expectedHist(numPartitions * numStats * totalBinFeatures, 0.0f);
    // Feature 0: firstFold=0
    expectedHist[0] = 7.0f;   // bin=1
    expectedHist[1] = 17.0f;  // bin=2
    // Feature 1: firstFold=2
    expectedHist[2] = 9.0f;   // bin=1
    expectedHist[3] = 12.0f;  // bin=2
    // Feature 2: firstFold=4
    expectedHist[4] = 7.0f;   // bin=1
    expectedHist[5] = 17.0f;  // bin=2
    // Feature 3: firstFold=6
    expectedHist[6] = 8.0f;   // bin=1
    expectedHist[7] = 13.0f;  // bin=2

    // --- GPU kernel dispatch ---
    // Build MLX arrays
    auto compressedData = mx::array(
        reinterpret_cast<const int32_t*>(packed.data()),
        {static_cast<int>(numDocs * numUi32PerDoc)},  // flat 1D
        mx::uint32
    );
    auto stats = mx::array(gradients.data(), {static_cast<int>(numDocs)}, mx::float32);

    // Single partition: all docs in order
    std::vector<uint32_t> docIndicesVec(numDocs);
    std::iota(docIndicesVec.begin(), docIndicesVec.end(), 0);
    auto docIndices = mx::array(
        reinterpret_cast<const int32_t*>(docIndicesVec.data()),
        {static_cast<int>(numDocs)}, mx::uint32
    );

    std::vector<uint32_t> partOffsetsVec = {0};
    std::vector<uint32_t> partSizesVec = {numDocs};
    auto partOffsets = mx::array(
        reinterpret_cast<const int32_t*>(partOffsetsVec.data()),
        {static_cast<int>(numPartitions)}, mx::uint32
    );
    auto partSizes = mx::array(
        reinterpret_cast<const int32_t*>(partSizesVec.data()),
        {static_cast<int>(numPartitions)}, mx::uint32
    );

    // Scalar constants
    auto featureColArr = mx::array(static_cast<uint32_t>(0), mx::uint32);  // group 0
    auto lineSizeArr   = mx::array(static_cast<uint32_t>(numUi32PerDoc), mx::uint32);
    auto maxBlocksArr  = mx::array(static_cast<uint32_t>(1), mx::uint32);
    auto totalBinsArr  = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
    auto numStatsArr   = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
    auto totalDocsArr  = mx::array(static_cast<uint32_t>(numDocs), mx::uint32);

    // Per-feature fold counts and first fold indices
    std::vector<uint32_t> foldCountsVec = {folds, folds, folds, folds};
    std::vector<uint32_t> firstFoldVec = {0, 2, 4, 6};
    auto foldCountsArr = mx::array(
        reinterpret_cast<const int32_t*>(foldCountsVec.data()),
        {4}, mx::uint32
    );
    auto firstFoldArr = mx::array(
        reinterpret_cast<const int32_t*>(firstFoldVec.data()),
        {4}, mx::uint32
    );

    // Register and dispatch kernel
    auto kernel = mx::fast::metal_kernel(
        "histogram_one_byte_features",
        /*input_names=*/{
            "compressedIndex", "stats", "docIndices",
            "partOffsets", "partSizes",
            "featureColumnIdx", "lineSize", "maxBlocksPerPart",
            "foldCounts", "firstFoldIndices",
            "totalBinFeatures", "numStats", "totalNumDocs"
        },
        /*output_names=*/{"histogram"},
        /*source=*/NCatboostMlx::KernelSources::kHistOneByteSource,
        /*header=*/NCatboostMlx::KernelSources::kHistHeader,
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false
    );

    mx::Shape histShape = {static_cast<int>(numPartitions * numStats * totalBinFeatures)};
    auto grid = std::make_tuple(256, 1, 1);  // 256 total threads = 1 threadgroup of 256
    auto threadgroup = std::make_tuple(256, 1, 1);

    auto results = kernel(
        /*inputs=*/{
            compressedData, stats, docIndices,
            partOffsets, partSizes,
            featureColArr, lineSizeArr, maxBlocksArr,
            foldCountsArr, firstFoldArr,
            totalBinsArr, numStatsArr, totalDocsArr
        },
        /*output_shapes=*/{histShape},
        /*output_dtypes=*/{mx::float32},
        grid,
        threadgroup,
        /*template_args=*/{},
        /*init_value=*/0.0f,
        /*verbose=*/false,
        /*stream=*/mx::Device::gpu
    );

    auto histogram = results[0];
    mx::eval(histogram);

    // --- Verify against CPU reference ---
    const float* histData = histogram.data<float>();
    for (uint32_t i = 0; i < totalBinFeatures; ++i) {
        EXPECT_NEAR(histData[i], expectedHist[i], 1e-5f)
            << "histogram[" << i << "] mismatch: GPU=" << histData[i]
            << " expected=" << expectedHist[i];
    }
}

TEST(MetalHistogram, KernelDispatchTwoPartitions) {
    // Test with 2 partitions to verify partition-aware histogram computation.
    //
    // 8 docs, 4 features, docs split into 2 partitions:
    //   Partition 0: docs {0, 2, 4, 6}  (sorted by partition)
    //   Partition 1: docs {1, 3, 5, 7}
    //
    // Feature bins (same as single-partition test):
    //   doc0: [0, 1, 0, 2]   (partition 0)
    //   doc1: [1, 0, 1, 0]   (partition 1)
    //   doc2: [2, 1, 2, 1]   (partition 0)
    //   doc3: [0, 2, 0, 2]   (partition 1)
    //   doc4: [1, 1, 1, 1]   (partition 0)
    //   doc5: [2, 0, 2, 0]   (partition 1)
    //   doc6: [0, 0, 0, 0]   (partition 0)
    //   doc7: [2, 2, 2, 2]   (partition 1)
    //
    // Gradients: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    const uint32_t numDocs = 8;
    const uint32_t numFeatures = 4;
    const uint32_t numUi32PerDoc = 1;
    const uint32_t folds = 2;
    const uint32_t totalBinFeatures = numFeatures * folds;  // 8
    const uint32_t numPartitions = 2;
    const uint32_t numStats = 1;

    std::vector<std::vector<uint8_t>> docs = {
        {0, 1, 0, 2},
        {1, 0, 1, 0},
        {2, 1, 2, 1},
        {0, 2, 0, 2},
        {1, 1, 1, 1},
        {2, 0, 2, 0},
        {0, 0, 0, 0},
        {2, 2, 2, 2}
    };
    auto packed = PackFeatures(docs);
    std::vector<float> gradients = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    // Sorted doc indices: partition 0 first, then partition 1
    std::vector<uint32_t> docIndicesVec = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<uint32_t> partOffsetsVec = {0, 4};
    std::vector<uint32_t> partSizesVec = {4, 4};

    // --- CPU reference ---
    // Partition 0 docs: 0(grads=1.0), 2(3.0), 4(5.0), 6(7.0)
    //
    // Feature 0 (part0 bins: 0,2,1,0):
    //   bin=1: doc4 -> 5.0
    //   bin=2: doc2 -> 3.0
    //
    // Feature 1 (part0 bins: 1,1,1,0):
    //   bin=1: doc0,doc2,doc4 -> 1.0+3.0+5.0 = 9.0
    //   bin=2: (none) -> 0.0
    //
    // Feature 2 (part0 bins: 0,2,1,0):
    //   bin=1: doc4 -> 5.0
    //   bin=2: doc2 -> 3.0
    //
    // Feature 3 (part0 bins: 2,1,1,0):
    //   bin=1: doc2,doc4 -> 3.0+5.0 = 8.0
    //   bin=2: doc0 -> 1.0

    // Partition 1 docs: 1(2.0), 3(4.0), 5(6.0), 7(8.0)
    //
    // Feature 0 (part1 bins: 1,0,2,2):
    //   bin=1: doc1 -> 2.0
    //   bin=2: doc5,doc7 -> 6.0+8.0 = 14.0
    //
    // Feature 1 (part1 bins: 0,2,0,2):
    //   bin=1: (none) -> 0.0
    //   bin=2: doc3,doc7 -> 4.0+8.0 = 12.0
    //
    // Feature 2 (part1 bins: 1,0,2,2):
    //   bin=1: doc1 -> 2.0
    //   bin=2: doc5,doc7 -> 6.0+8.0 = 14.0
    //
    // Feature 3 (part1 bins: 0,2,0,2):
    //   bin=1: (none) -> 0.0
    //   bin=2: doc3,doc7 -> 4.0+8.0 = 12.0

    // Layout: histogram[part * numStats * totalBinFeatures + stat * totalBinFeatures + binFeature]
    std::vector<float> expectedHist(numPartitions * numStats * totalBinFeatures, 0.0f);
    // Partition 0 (offset 0)
    expectedHist[0] = 5.0f;   // feat0, bin1
    expectedHist[1] = 3.0f;   // feat0, bin2
    expectedHist[2] = 9.0f;   // feat1, bin1
    expectedHist[3] = 0.0f;   // feat1, bin2
    expectedHist[4] = 5.0f;   // feat2, bin1
    expectedHist[5] = 3.0f;   // feat2, bin2
    expectedHist[6] = 8.0f;   // feat3, bin1
    expectedHist[7] = 1.0f;   // feat3, bin2
    // Partition 1 (offset 8)
    expectedHist[8]  = 2.0f;   // feat0, bin1
    expectedHist[9]  = 14.0f;  // feat0, bin2
    expectedHist[10] = 0.0f;   // feat1, bin1
    expectedHist[11] = 12.0f;  // feat1, bin2
    expectedHist[12] = 2.0f;   // feat2, bin1
    expectedHist[13] = 14.0f;  // feat2, bin2
    expectedHist[14] = 0.0f;   // feat3, bin1
    expectedHist[15] = 12.0f;  // feat3, bin2

    // --- GPU dispatch ---
    auto compressedData = mx::array(
        reinterpret_cast<const int32_t*>(packed.data()),
        {static_cast<int>(numDocs * numUi32PerDoc)},
        mx::uint32
    );
    auto stats = mx::array(gradients.data(), {static_cast<int>(numDocs)}, mx::float32);
    auto docIndices = mx::array(
        reinterpret_cast<const int32_t*>(docIndicesVec.data()),
        {static_cast<int>(numDocs)}, mx::uint32
    );
    auto partOffsets = mx::array(
        reinterpret_cast<const int32_t*>(partOffsetsVec.data()),
        {static_cast<int>(numPartitions)}, mx::uint32
    );
    auto partSizesArr = mx::array(
        reinterpret_cast<const int32_t*>(partSizesVec.data()),
        {static_cast<int>(numPartitions)}, mx::uint32
    );

    auto featureColArr = mx::array(static_cast<uint32_t>(0), mx::uint32);
    auto lineSizeArr   = mx::array(static_cast<uint32_t>(numUi32PerDoc), mx::uint32);
    auto maxBlocksArr  = mx::array(static_cast<uint32_t>(1), mx::uint32);
    auto totalBinsArr  = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
    auto numStatsArr   = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
    auto totalDocsArr  = mx::array(static_cast<uint32_t>(numDocs), mx::uint32);

    std::vector<uint32_t> foldCountsVec = {folds, folds, folds, folds};
    std::vector<uint32_t> firstFoldVec = {0, 2, 4, 6};
    auto foldCountsArr = mx::array(
        reinterpret_cast<const int32_t*>(foldCountsVec.data()),
        {4}, mx::uint32
    );
    auto firstFoldArr = mx::array(
        reinterpret_cast<const int32_t*>(firstFoldVec.data()),
        {4}, mx::uint32
    );

    auto kernel = mx::fast::metal_kernel(
        "histogram_one_byte_features",
        {"compressedIndex", "stats", "docIndices",
         "partOffsets", "partSizes",
         "featureColumnIdx", "lineSize", "maxBlocksPerPart",
         "foldCounts", "firstFoldIndices",
         "totalBinFeatures", "numStats", "totalNumDocs"},
        {"histogram"},
        NCatboostMlx::KernelSources::kHistOneByteSource,
        NCatboostMlx::KernelSources::kHistHeader,
        true,
        false
    );

    mx::Shape histShape = {static_cast<int>(numPartitions * numStats * totalBinFeatures)};
    // Grid: (256 * blocks, partitions, stats) — total threads, not threadgroup count
    auto grid = std::make_tuple(256, static_cast<int>(numPartitions), 1);
    auto threadgroup = std::make_tuple(256, 1, 1);

    auto results = kernel(
        {compressedData, stats, docIndices,
         partOffsets, partSizesArr,
         featureColArr, lineSizeArr, maxBlocksArr,
         foldCountsArr, firstFoldArr,
         totalBinsArr, numStatsArr, totalDocsArr},
        {histShape},
        {mx::float32},
        grid,
        threadgroup,
        {},
        0.0f,
        false,
        mx::Device::gpu
    );

    auto histogram = results[0];
    mx::eval(histogram);

    const float* histData = histogram.data<float>();
    for (uint32_t i = 0; i < numPartitions * numStats * totalBinFeatures; ++i) {
        EXPECT_NEAR(histData[i], expectedHist[i], 1e-5f)
            << "histogram[" << i << "] mismatch: GPU=" << histData[i]
            << " expected=" << expectedHist[i];
    }
}

TEST(MetalHistogram, KernelDispatchWithWeights) {
    // Test with numStats=2 (gradient + hessian), single partition.
    // This validates the 2-stat channel layout used by the real training loop.
    //
    // 4 docs, 2 features (bins: 0 or 1), folds=1 each
    // Gradients: [1.0, 2.0, 3.0, 4.0]
    // Hessians:  [1.0, 1.0, 1.0, 1.0]  (RMSE)

    const uint32_t numDocs = 4;
    const uint32_t numFeatures = 2;
    const uint32_t numUi32PerDoc = 1;
    const uint32_t totalBinFeatures = 2;  // 2 features * 1 fold each
    const uint32_t numPartitions = 1;
    const uint32_t numStats = 2;

    // Feature bins: doc0=[0,1], doc1=[1,0], doc2=[0,1], doc3=[1,0]
    std::vector<std::vector<uint8_t>> docs = {
        {0, 1},
        {1, 0},
        {0, 1},
        {1, 0}
    };
    auto packed = PackFeatures(docs);

    // Stats array: [grads..., hessians...] = [numStats * numDocs]
    std::vector<float> statsVec = {
        1.0f, 2.0f, 3.0f, 4.0f,   // gradients
        1.0f, 1.0f, 1.0f, 1.0f    // hessians
    };

    // CPU reference:
    // Feature 0 (bins: 0,1,0,1), folds=1, firstFold=0:
    //   bin=1 gradient: docs 1,3 -> 2.0+4.0 = 6.0
    //   bin=1 hessian:  docs 1,3 -> 1.0+1.0 = 2.0
    //
    // Feature 1 (bins: 1,0,1,0), folds=1, firstFold=1:
    //   bin=1 gradient: docs 0,2 -> 1.0+3.0 = 4.0
    //   bin=1 hessian:  docs 0,2 -> 1.0+1.0 = 2.0
    //
    // Layout: [part0_stat0_bins..., part0_stat1_bins...]
    // = [gradHist[0], gradHist[1], hessHist[0], hessHist[1]]
    std::vector<float> expectedHist = {
        6.0f, 4.0f,   // stat=0 (gradient): feat0_bin1, feat1_bin1
        2.0f, 2.0f    // stat=1 (hessian):  feat0_bin1, feat1_bin1
    };

    // Build MLX arrays
    auto compressedData = mx::array(
        reinterpret_cast<const int32_t*>(packed.data()),
        {static_cast<int>(numDocs * numUi32PerDoc)},
        mx::uint32
    );
    auto stats = mx::array(statsVec.data(), {static_cast<int>(numStats * numDocs)}, mx::float32);

    std::vector<uint32_t> docIndicesVec = {0, 1, 2, 3};
    auto docIndices = mx::array(
        reinterpret_cast<const int32_t*>(docIndicesVec.data()),
        {static_cast<int>(numDocs)}, mx::uint32
    );

    std::vector<uint32_t> partOffsetsVec = {0};
    std::vector<uint32_t> partSizesVec = {numDocs};
    auto partOffsets = mx::array(
        reinterpret_cast<const int32_t*>(partOffsetsVec.data()), {1}, mx::uint32);
    auto partSizesArr = mx::array(
        reinterpret_cast<const int32_t*>(partSizesVec.data()), {1}, mx::uint32);

    auto featureColArr = mx::array(static_cast<uint32_t>(0), mx::uint32);
    auto lineSizeArr   = mx::array(static_cast<uint32_t>(numUi32PerDoc), mx::uint32);
    auto maxBlocksArr  = mx::array(static_cast<uint32_t>(1), mx::uint32);
    auto totalBinsArr  = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
    auto numStatsArr   = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
    auto totalDocsArr  = mx::array(static_cast<uint32_t>(numDocs), mx::uint32);

    // Feature 0: 2 bins -> folds=1; Feature 1: 2 bins -> folds=1
    // Pad to 4 features for kernel (remaining features have folds=0)
    std::vector<uint32_t> foldCountsVec = {1, 1, 0, 0};
    std::vector<uint32_t> firstFoldVec = {0, 1, 0, 0};
    auto foldCountsArr = mx::array(
        reinterpret_cast<const int32_t*>(foldCountsVec.data()), {4}, mx::uint32);
    auto firstFoldArr = mx::array(
        reinterpret_cast<const int32_t*>(firstFoldVec.data()), {4}, mx::uint32);

    auto kernel = mx::fast::metal_kernel(
        "histogram_one_byte_features",
        {"compressedIndex", "stats", "docIndices",
         "partOffsets", "partSizes",
         "featureColumnIdx", "lineSize", "maxBlocksPerPart",
         "foldCounts", "firstFoldIndices",
         "totalBinFeatures", "numStats", "totalNumDocs"},
        {"histogram"},
        NCatboostMlx::KernelSources::kHistOneByteSource,
        NCatboostMlx::KernelSources::kHistHeader,
        true, false
    );

    mx::Shape histShape = {static_cast<int>(numPartitions * numStats * totalBinFeatures)};
    // Grid: (256 * blocks, partitions, stats) — total threads
    auto grid = std::make_tuple(256, 1, static_cast<int>(numStats));
    auto threadgroup = std::make_tuple(256, 1, 1);

    auto results = kernel(
        {compressedData, stats, docIndices,
         partOffsets, partSizesArr,
         featureColArr, lineSizeArr, maxBlocksArr,
         foldCountsArr, firstFoldArr,
         totalBinsArr, numStatsArr, totalDocsArr},
        {histShape},
        {mx::float32},
        grid, threadgroup,
        {}, 0.0f, false, mx::Device::gpu
    );

    auto histogram = results[0];
    mx::eval(histogram);

    const float* histData = histogram.data<float>();
    for (uint32_t i = 0; i < numPartitions * numStats * totalBinFeatures; ++i) {
        EXPECT_NEAR(histData[i], expectedHist[i], 1e-5f)
            << "histogram[" << i << "] mismatch: GPU=" << histData[i]
            << " expected=" << expectedHist[i];
    }
}
