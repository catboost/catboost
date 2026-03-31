// Standalone Metal kernel dispatch verification.
// Compiles with: clang++ -std=c++17 -I/opt/homebrew/Cellar/mlx/0.31.1/include
//                -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx
//                -framework Metal -framework Foundation
//                catboost/mlx/tests/standalone_kernel_test.cpp -o /tmp/kernel_test && /tmp/kernel_test

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <numeric>
#include <cmath>
#include <cassert>
#include <string>

namespace mx = mlx::core;

// ---- Kernel source strings (copied from kernel_sources.h) ----

static const std::string kHistHeader = R"metal(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint FEATURES_PER_PACK = 4;
constant constexpr uint BINS_PER_BYTE = 256;
constant constexpr uint BLOCK_SIZE = 256;
constant constexpr uint NUM_SIMD_GROUPS = BLOCK_SIZE / SIMD_SIZE;
constant constexpr uint HIST_PER_SIMD = FEATURES_PER_PACK * BINS_PER_BYTE;
constant constexpr uint TOTAL_HIST_SIZE = NUM_SIMD_GROUPS * HIST_PER_SIMD;
)metal";

static const std::string kHistOneByteSource = R"metal(
    const uint partIdx   = threadgroup_position_in_grid.y;
    const uint statIdx   = threadgroup_position_in_grid.z;
    const uint blockInPart = threadgroup_position_in_grid.x;

    const uint partOffset = partOffsets[partIdx];
    const uint partSize   = partSizes[partIdx];

    if (partSize == 0) return;

    const uint docsPerBlock = (partSize + maxBlocksPerPart - 1) / maxBlocksPerPart;
    const uint myDocStart = blockInPart * docsPerBlock;
    if (myDocStart >= partSize) return;
    const uint myDocEnd = min(myDocStart + docsPerBlock, partSize);
    const uint myDocCount = myDocEnd - myDocStart;

    // CAS-based float atomic add on threadgroup atomic_uint.
    // Metal supports atomic_uint in threadgroup but NOT atomic_float.
    threadgroup atomic_uint sharedHist[HIST_PER_SIMD];

    for (uint i = thread_index_in_threadgroup; i < HIST_PER_SIMD; i += BLOCK_SIZE) {
        atomic_store_explicit(&sharedHist[i], as_type<uint>(0.0f), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = thread_index_in_threadgroup; d < myDocCount; d += BLOCK_SIZE) {
        const uint sortedPos = partOffset + myDocStart + d;
        const uint docIdx = docIndices[sortedPos];

        const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];
        const float stat = stats[statIdx * totalNumDocs + docIdx];

        for (uint f = 0; f < FEATURES_PER_PACK; f++) {
            const uint bin = (packed >> (24 - 8 * f)) & 0xFF;
            if (bin < foldCounts[f] + 1) {
                const uint histIdx = f * BINS_PER_BYTE + bin;
                // CAS-based float atomic add
                uint old_val = atomic_load_explicit(&sharedHist[histIdx], memory_order_relaxed);
                uint new_val;
                do {
                    new_val = as_type<uint>(as_type<float>(old_val) + stat);
                } while (!atomic_compare_exchange_weak_explicit(
                    &sharedHist[histIdx], &old_val, new_val,
                    memory_order_relaxed, memory_order_relaxed));
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint histBase = partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures;

    for (uint f = 0; f < FEATURES_PER_PACK; f++) {
        const uint folds = foldCounts[f];
        const uint firstFold = firstFoldIndices[f];

        for (uint bin = thread_index_in_threadgroup; bin < folds; bin += BLOCK_SIZE) {
            const float val = as_type<float>(atomic_load_explicit(&sharedHist[f * BINS_PER_BYTE + bin + 1], memory_order_relaxed));
            if (abs(val) > 1e-20f) {
                if (maxBlocksPerPart > 1) {
                    device atomic_float* dst = (device atomic_float*)(histogram + histBase + firstFold + bin);
                    atomic_fetch_add_explicit(dst, val, memory_order_relaxed);
                } else {
                    histogram[histBase + firstFold + bin] = val;
                }
            }
        }
    }
)metal";

// ---- Pack helper ----

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

// ---- Test: single partition histogram ----

bool TestSinglePartition() {
    printf("=== Test: Single Partition Histogram ===\n");

    const uint32_t numDocs = 8;
    const uint32_t numUi32PerDoc = 1;
    const uint32_t folds = 2;
    const uint32_t totalBinFeatures = 4 * folds;  // 8
    const uint32_t numPartitions = 1;
    const uint32_t numStats = 1;

    std::vector<std::vector<uint8_t>> docs = {
        {0, 1, 0, 2}, {1, 0, 1, 0}, {2, 1, 2, 1}, {0, 2, 0, 2},
        {1, 1, 1, 1}, {2, 0, 2, 0}, {0, 0, 0, 0}, {2, 2, 2, 2}
    };
    auto packed = PackFeatures(docs);
    std::vector<float> gradients = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    // Expected: histogram[firstFold+bin] = sum of grads for docs where feature_bin == bin+1
    float expected[] = {
        7.0f, 17.0f,  // feat0: bin1=7, bin2=17
        9.0f, 12.0f,  // feat1: bin1=9, bin2=12
        7.0f, 17.0f,  // feat2: bin1=7, bin2=17
        8.0f, 13.0f   // feat3: bin1=8, bin2=13
    };

    // Build MLX arrays
    auto compressedData = mx::array(
        reinterpret_cast<const int32_t*>(packed.data()),
        {static_cast<int>(numDocs * numUi32PerDoc)}, mx::uint32);
    auto stats = mx::array(gradients.data(), {static_cast<int>(numDocs)}, mx::float32);

    std::vector<uint32_t> docIndicesVec(numDocs);
    std::iota(docIndicesVec.begin(), docIndicesVec.end(), 0);
    auto docIndices = mx::array(
        reinterpret_cast<const int32_t*>(docIndicesVec.data()),
        {static_cast<int>(numDocs)}, mx::uint32);

    uint32_t partOff = 0, partSz = numDocs;
    auto partOffsets = mx::array(reinterpret_cast<const int32_t*>(&partOff), {1}, mx::uint32);
    auto partSizes = mx::array(reinterpret_cast<const int32_t*>(&partSz), {1}, mx::uint32);

    auto featureColArr = mx::array(static_cast<uint32_t>(0), mx::uint32);
    auto lineSizeArr   = mx::array(static_cast<uint32_t>(numUi32PerDoc), mx::uint32);
    auto maxBlocksArr  = mx::array(static_cast<uint32_t>(1), mx::uint32);
    auto totalBinsArr  = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
    auto numStatsArr   = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
    auto totalDocsArr  = mx::array(static_cast<uint32_t>(numDocs), mx::uint32);

    uint32_t foldCountsData[] = {folds, folds, folds, folds};
    uint32_t firstFoldData[] = {0, 2, 4, 6};
    auto foldCountsArr = mx::array(
        reinterpret_cast<const int32_t*>(foldCountsData), {4}, mx::uint32);
    auto firstFoldArr = mx::array(
        reinterpret_cast<const int32_t*>(firstFoldData), {4}, mx::uint32);

    // Register kernel
    printf("  Registering Metal kernel...\n");
    auto kernel = mx::fast::metal_kernel(
        "histogram_one_byte_features",
        {"compressedIndex", "stats", "docIndices",
         "partOffsets", "partSizes",
         "featureColumnIdx", "lineSize", "maxBlocksPerPart",
         "foldCounts", "firstFoldIndices",
         "totalBinFeatures", "numStats", "totalNumDocs"},
        {"histogram"},
        kHistOneByteSource,
        kHistHeader,
        true,   // ensure_row_contiguous
        false   // atomic_outputs
    );

    printf("  Dispatching kernel...\n");
    mx::Shape histShape = {static_cast<int>(totalBinFeatures)};
    auto results = kernel(
        {compressedData, stats, docIndices,
         partOffsets, partSizes,
         featureColArr, lineSizeArr, maxBlocksArr,
         foldCountsArr, firstFoldArr,
         totalBinsArr, numStatsArr, totalDocsArr},
        {histShape}, {mx::float32},
        std::make_tuple(256, 1, 1),    // grid = (threads_x, partitions, stats)
        std::make_tuple(256, 1, 1),    // threadgroup
        {},       // template_args
        0.0f,     // init_value
        false,    // verbose
        mx::Device::gpu
    );

    auto histogram = results[0];
    mx::eval(histogram);

    const float* histData = histogram.data<float>();
    bool pass = true;
    for (int i = 0; i < static_cast<int>(totalBinFeatures); ++i) {
        if (std::abs(histData[i] - expected[i]) > 1e-4f) {
            printf("  FAIL: histogram[%d] = %.4f, expected %.4f\n", i, histData[i], expected[i]);
            pass = false;
        }
    }

    if (pass) {
        printf("  PASS: All %d histogram bins match expected values.\n", totalBinFeatures);
    }
    return pass;
}

// ---- Test: two partitions ----

bool TestTwoPartitions() {
    printf("\n=== Test: Two Partition Histogram ===\n");

    const uint32_t numDocs = 8;
    const uint32_t numUi32PerDoc = 1;
    const uint32_t folds = 2;
    const uint32_t totalBinFeatures = 4 * folds;
    const uint32_t numPartitions = 2;
    const uint32_t numStats = 1;

    std::vector<std::vector<uint8_t>> docs = {
        {0, 1, 0, 2}, {1, 0, 1, 0}, {2, 1, 2, 1}, {0, 2, 0, 2},
        {1, 1, 1, 1}, {2, 0, 2, 0}, {0, 0, 0, 0}, {2, 2, 2, 2}
    };
    auto packed = PackFeatures(docs);
    std::vector<float> gradients = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    // Partition 0: docs {0,2,4,6}, Partition 1: docs {1,3,5,7}
    uint32_t docIndicesData[] = {0, 2, 4, 6, 1, 3, 5, 7};
    uint32_t partOffsetsData[] = {0, 4};
    uint32_t partSizesData[] = {4, 4};

    float expected[] = {
        // Part 0: feat0(bin1=5,bin2=3) feat1(bin1=9,bin2=0) feat2(bin1=5,bin2=3) feat3(bin1=8,bin2=1)
        5.0f, 3.0f, 9.0f, 0.0f, 5.0f, 3.0f, 8.0f, 1.0f,
        // Part 1: feat0(bin1=2,bin2=14) feat1(bin1=0,bin2=12) feat2(bin1=2,bin2=14) feat3(bin1=0,bin2=12)
        2.0f, 14.0f, 0.0f, 12.0f, 2.0f, 14.0f, 0.0f, 12.0f
    };

    auto compressedData = mx::array(
        reinterpret_cast<const int32_t*>(packed.data()),
        {static_cast<int>(numDocs * numUi32PerDoc)}, mx::uint32);
    auto stats = mx::array(gradients.data(), {static_cast<int>(numDocs)}, mx::float32);
    auto docIndices = mx::array(
        reinterpret_cast<const int32_t*>(docIndicesData),
        {static_cast<int>(numDocs)}, mx::uint32);
    auto partOffsets = mx::array(
        reinterpret_cast<const int32_t*>(partOffsetsData), {2}, mx::uint32);
    auto partSizes = mx::array(
        reinterpret_cast<const int32_t*>(partSizesData), {2}, mx::uint32);

    auto featureColArr = mx::array(static_cast<uint32_t>(0), mx::uint32);
    auto lineSizeArr   = mx::array(static_cast<uint32_t>(numUi32PerDoc), mx::uint32);
    auto maxBlocksArr  = mx::array(static_cast<uint32_t>(1), mx::uint32);
    auto totalBinsArr  = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
    auto numStatsArr   = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
    auto totalDocsArr  = mx::array(static_cast<uint32_t>(numDocs), mx::uint32);

    uint32_t foldCountsData[] = {folds, folds, folds, folds};
    uint32_t firstFoldData[] = {0, 2, 4, 6};
    auto foldCountsArr = mx::array(
        reinterpret_cast<const int32_t*>(foldCountsData), {4}, mx::uint32);
    auto firstFoldArr = mx::array(
        reinterpret_cast<const int32_t*>(firstFoldData), {4}, mx::uint32);

    printf("  Dispatching kernel...\n");
    auto kernel = mx::fast::metal_kernel(
        "histogram_one_byte_features",
        {"compressedIndex", "stats", "docIndices",
         "partOffsets", "partSizes",
         "featureColumnIdx", "lineSize", "maxBlocksPerPart",
         "foldCounts", "firstFoldIndices",
         "totalBinFeatures", "numStats", "totalNumDocs"},
        {"histogram"},
        kHistOneByteSource, kHistHeader, true, false
    );

    const int histSize = static_cast<int>(numPartitions * numStats * totalBinFeatures);
    auto results = kernel(
        {compressedData, stats, docIndices,
         partOffsets, partSizes,
         featureColArr, lineSizeArr, maxBlocksArr,
         foldCountsArr, firstFoldArr,
         totalBinsArr, numStatsArr, totalDocsArr},
        {{histSize}}, {mx::float32},
        std::make_tuple(256, static_cast<int>(numPartitions), 1),  // grid = (threads_x, partitions, stats)
        std::make_tuple(256, 1, 1),  // threadgroup
        {}, 0.0f, false, mx::Device::gpu
    );

    auto histogram = results[0];
    mx::eval(histogram);

    const float* histData = histogram.data<float>();
    bool pass = true;
    for (int i = 0; i < histSize; ++i) {
        if (std::abs(histData[i] - expected[i]) > 1e-4f) {
            printf("  FAIL: histogram[%d] = %.4f, expected %.4f\n", i, histData[i], expected[i]);
            pass = false;
        }
    }

    if (pass) {
        printf("  PASS: All %d histogram bins match expected values.\n", histSize);
    }
    return pass;
}

int main() {
    printf("CatBoost-MLX Metal Kernel Dispatch Test\n");
    printf("========================================\n\n");

    bool allPass = true;
    allPass &= TestSinglePartition();
    allPass &= TestTwoPartitions();

    printf("\n========================================\n");
    if (allPass) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}
