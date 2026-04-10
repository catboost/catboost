// bench_boosting.cpp — Library-path benchmark harness for CatBoost-MLX (TODO-014)
//
// PURPOSE
//   Synthesizes a realistic in-memory dataset and exercises the full GBDT
//   pipeline through the production Metal kernels (histogram, suffix-sum,
//   split scoring, leaf accumulation, tree application) — the same code path
//   that RunBoosting / SearchTreeStructure call. Measures per-iteration wall
//   time so future sprints can quantify speedups.
//
//   This binary does NOT call csv_train or any subprocess. It is a direct
//   exercise of the kernel dispatch logic that the library API uses.
//
// USAGE
//   ./bench_boosting [options]
//     --rows N          Number of training documents (default: 100000)
//     --features N      Number of features (default: 50)
//     --classes N       Number of classes: 1 = regression, 2 = binary, K >= 3 = multiclass (default: 1)
//     --depth D         Max tree depth (default: 6)
//     --iters N         Number of boosting iterations (default: 100)
//     --bins B          Bins per feature (default: 32; max 255)
//     --lr F            Learning rate (default: 0.1)
//     --l2 F            L2 regularization lambda (default: 3.0)
//     --seed N          Random seed for data synthesis (default: 42)
//     --onehot N        Mark the first N features as one-hot encoded (default: 0)
//
// OUTPUT
//   Prints per-iteration timings, a summary table (iter-0 cold-start vs warm
//   average), and the final training loss for regression testing.
//
// COMPILE (standalone, no CatBoost headers required)
//   clang++ -std=c++17 -O2 -I. \
//     -I/opt/homebrew/Cellar/mlx/0.31.1/include \
//     -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
//     -framework Metal -framework Foundation -Wno-c++20-extensions \
//     catboost/mlx/tests/bench_boosting.cpp -o bench_boosting
//
// EXAMPLES
//   # Binary classification: 100k rows, 50 features, depth 6, 100 iterations
//   ./bench_boosting --rows 100000 --features 50 --classes 2 --depth 6 --iters 100
//
//   # Multiclass K=3: 20k rows, 30 features, depth 5, 50 iterations
//   ./bench_boosting --rows 20000 --features 30 --classes 3 --depth 5 --iters 50
//
//   # Regression baseline
//   ./bench_boosting --rows 100000 --features 50 --classes 1 --depth 6 --iters 100
//
//   # Exercise the one-hot skip branch in kSuffixSumSource (5 one-hot features)
//   ./bench_boosting --rows 10000 --features 20 --classes 2 --depth 4 --iters 30 \
//                    --bins 32 --seed 42 --onehot 5
//
// WHAT THIS MEASURES
//   - Per-iteration wall time: histogram build + suffix-sum + split scoring +
//     leaf accumulation + tree application
//   - iter-0 is Metal's cold-start (kernel compile + first dispatch)
//   - warm average = mean of iters 1..N-1
//
// WHAT THIS DOES NOT MEASURE
//   - Python overhead (bindings → subprocess)
//   - CatBoost feature quantization (data is pre-quantized synthetically)
//   - Model export or prediction

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <catboost/mlx/kernels/kernel_sources.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <cassert>
#include <limits>

namespace mx = mlx::core;
using ui32 = uint32_t;
using ui64 = uint64_t;

namespace NCatboostMlx {
namespace KernelSources {}  // sourced via #include above
}

using namespace NCatboostMlx;

// ============================================================================
// Mirrored structures (same layout as gpu_structures.h, no CatBoost headers)
// ============================================================================

struct TCFeature {
    ui64 Offset;
    ui32 Mask;
    ui32 Shift;
    ui32 FirstFoldIndex;
    ui32 Folds;
    bool OneHotFeature;
    bool SkipFirstBinInScoreCount;
};

struct TBestSplitProperties {
    ui32 FeatureId = static_cast<ui32>(-1);
    ui32 BinId = 0;
    float Score = std::numeric_limits<float>::infinity();
    float Gain = std::numeric_limits<float>::infinity();
    bool Defined() const { return FeatureId != static_cast<ui32>(-1); }
};

// ============================================================================
// CLI argument parsing
// ============================================================================

struct TBenchConfig {
    ui32 NumRows     = 100000;
    ui32 NumFeatures = 50;
    ui32 NumClasses  = 1;     // 1=regression, 2=binary, >=3=multiclass
    ui32 MaxDepth    = 6;
    ui32 NumIters    = 100;
    ui32 NumBins     = 32;    // bins per feature (max 255)
    float LearningRate = 0.1f;
    float L2RegLambda  = 3.0f;
    ui64 Seed          = 42;
    ui32 NumOneHot     = 0;   // first N features treated as one-hot encoded
};

TBenchConfig ParseArgs(int argc, char** argv) {
    TBenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--rows")     == 0 && i+1 < argc) cfg.NumRows     = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--features") == 0 && i+1 < argc) cfg.NumFeatures = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--classes")  == 0 && i+1 < argc) cfg.NumClasses  = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth")    == 0 && i+1 < argc) cfg.MaxDepth    = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters")    == 0 && i+1 < argc) cfg.NumIters    = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--bins")     == 0 && i+1 < argc) cfg.NumBins     = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr")       == 0 && i+1 < argc) cfg.LearningRate= std::atof(argv[++i]);
        else if (strcmp(argv[i], "--l2")       == 0 && i+1 < argc) cfg.L2RegLambda = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--seed")     == 0 && i+1 < argc) cfg.Seed        = std::atoll(argv[++i]);
        else if (strcmp(argv[i], "--onehot")   == 0 && i+1 < argc) cfg.NumOneHot   = std::atoi(argv[++i]);
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr,
                "Usage: bench_boosting [--rows N] [--features N] [--classes N]\n"
                "                      [--depth D] [--iters N] [--bins B]\n"
                "                      [--lr F] [--l2 F] [--seed N] [--onehot N]\n");
            exit(1);
        }
    }
    // Clamp bins to valid range
    if (cfg.NumBins < 2)   cfg.NumBins = 2;
    if (cfg.NumBins > 255) cfg.NumBins = 255;
    // approxDim: for multiclass K classes → K-1 dims; otherwise 1
    return cfg;
}

// ============================================================================
// Synthetic dataset generation
// ============================================================================
// Generates:
//   compressedData: [numRows, numUi32PerDoc] uint32
//     Each doc packs 4 features per uint32 (one-byte feature layout).
//   targets: [numRows] float32
//     Regression: continuous. Binary: 0/1. Multiclass: 0..K-1.
//   features: TCFeature metadata for the kernel (Folds, FirstFoldIndex, etc.)

struct TSynthDataset {
    std::vector<uint32_t>  CompressedData;   // [numRows * numUi32PerDoc]
    std::vector<float>     Targets;          // [numRows]
    std::vector<TCFeature> Features;
    ui32 NumRows;
    ui32 NumFeatures;
    ui32 NumUi32PerDoc;
    ui32 TotalBinFeatures;
};

TSynthDataset GenerateSynthDataset(const TBenchConfig& cfg) {
    TSynthDataset ds;
    ds.NumRows     = cfg.NumRows;
    ds.NumFeatures = cfg.NumFeatures;
    // Pack 4 features per uint32
    ds.NumUi32PerDoc = (cfg.NumFeatures + 3) / 4;

    // Clamp NumOneHot to the actual feature count
    const ui32 numOneHot = std::min(cfg.NumOneHot, cfg.NumFeatures);

    std::mt19937_64 rng(cfg.Seed);

    // Build feature metadata.
    // One-hot features (first numOneHot): small bin count in [2, 10], OneHotFeature=true.
    // Ordinal features (remainder):       bin count = cfg.NumBins, OneHotFeature=false.
    ui32 totalBinFeatures = 0;
    ds.Features.resize(cfg.NumFeatures);
    for (ui32 f = 0; f < cfg.NumFeatures; ++f) {
        const bool isOneHot = (f < numOneHot);
        // One-hot bin count: derive a small value (2–10) deterministically from f and seed
        // so different features get different sizes, and the result is reproducible.
        const ui32 folds = isOneHot
            ? static_cast<ui32>(2 + ((cfg.Seed + f) % 9))  // 2..10
            : cfg.NumBins;

        ds.Features[f].Offset         = f / 4;
        ds.Features[f].Shift          = 24 - 8 * (f % 4);
        ds.Features[f].Mask           = 0xFF;
        ds.Features[f].FirstFoldIndex = totalBinFeatures;
        ds.Features[f].Folds          = folds;
        ds.Features[f].OneHotFeature  = isOneHot;
        ds.Features[f].SkipFirstBinInScoreCount = false;
        totalBinFeatures += folds;
    }
    ds.TotalBinFeatures = totalBinFeatures;

    // Allocate compressed data
    ds.CompressedData.assign(cfg.NumRows * ds.NumUi32PerDoc, 0u);

    // Generate random bin values; respect each feature's actual bin count
    for (ui32 d = 0; d < cfg.NumRows; ++d) {
        for (ui32 f = 0; f < cfg.NumFeatures; ++f) {
            const ui32 folds = ds.Features[f].Folds;
            std::uniform_int_distribution<int> binDist(0, static_cast<int>(folds) - 1);
            uint8_t bin = static_cast<uint8_t>(binDist(rng));
            ui32 wordIdx = f / 4;
            ui32 shift   = 24 - 8 * (f % 4);
            ds.CompressedData[d * ds.NumUi32PerDoc + wordIdx] |= (static_cast<ui32>(bin) << shift);
        }
    }

    // Generate targets
    ds.Targets.resize(cfg.NumRows);
    if (cfg.NumClasses == 1) {
        // Regression: linear combination of first few features
        std::uniform_real_distribution<float> noiseDist(-0.1f, 0.1f);
        for (ui32 d = 0; d < cfg.NumRows; ++d) {
            float val = 0.0f;
            for (ui32 f = 0; f < std::min(cfg.NumFeatures, 5u); ++f) {
                ui32 wordIdx = f / 4;
                ui32 shift   = 24 - 8 * (f % 4);
                uint8_t bin  = (ds.CompressedData[d * ds.NumUi32PerDoc + wordIdx] >> shift) & 0xFF;
                val += static_cast<float>(bin) / cfg.NumBins * (f % 2 == 0 ? 1.0f : -0.5f);
            }
            ds.Targets[d] = val + noiseDist(rng);
        }
    } else {
        // Classification: assign class based on dominant feature
        for (ui32 d = 0; d < cfg.NumRows; ++d) {
            uint8_t bin0 = (ds.CompressedData[d * ds.NumUi32PerDoc] >> 24) & 0xFF;
            ds.Targets[d] = static_cast<float>(bin0 % cfg.NumClasses);
        }
    }

    return ds;
}

// ============================================================================
// Partition layout (GPU bucket sort — mirrors structure_searcher.cpp)
// ============================================================================

struct TPartitionLayout {
    mx::array DocIndices;   // [numDocs] uint32
    mx::array PartOffsets;  // [numParts] uint32
    mx::array PartSizes;    // [numParts] uint32
};

TPartitionLayout ComputePartitionLayout(
    const mx::array& partitions, ui32 numDocs, ui32 numPartitions)
{
    auto docIndices = mx::astype(mx::argsort(partitions, 0), mx::uint32);

    auto onesF = mx::ones({static_cast<int>(numDocs)}, mx::float32);
    auto partSizesF = mx::scatter_add_axis(
        mx::zeros({static_cast<int>(numPartitions)}, mx::float32),
        partitions, onesF, 0);
    auto partOffsetsF = mx::subtract(mx::cumsum(partSizesF, 0), partSizesF);

    return TPartitionLayout{
        .DocIndices  = docIndices,
        .PartOffsets = mx::astype(partOffsetsF, mx::uint32),
        .PartSizes   = mx::astype(partSizesF,   mx::uint32)
    };
}

// ============================================================================
// Histogram dispatch (mirrors histogram.cpp batched dispatch)
// ============================================================================

mx::array DispatchHistogram(
    const mx::array& compressedData,
    const mx::array& stats,           // [numDocs] for single stat
    const mx::array& docIndices,
    const mx::array& partOffsets,
    const mx::array& partSizes,
    const std::vector<TCFeature>& features,
    ui32 numUi32PerDoc,
    ui32 numPartitions,
    ui32 totalBinFeatures,
    ui32 numStats,
    ui32 numDocs,
    ui32 maxBlocksPerPart
) {
    const ui32 numFeatures = static_cast<ui32>(features.size());
    // Group features into batches of 4 (one uint32 column per group)
    const ui32 numGroups = (numFeatures + 3) / 4;

    // Build flat fold metadata for all groups (4 entries per group)
    std::vector<uint32_t> foldCountsFlat(numGroups * 4, 0u);
    std::vector<uint32_t> firstFoldIndicesFlat(numGroups * 4, 0u);
    std::vector<uint32_t> featureColumnIndicesVec(numGroups, 0u);

    for (ui32 g = 0; g < numGroups; ++g) {
        featureColumnIndicesVec[g] = g;  // each group maps to column g
        for (ui32 slot = 0; slot < 4; ++slot) {
            ui32 f = g * 4 + slot;
            if (f < numFeatures) {
                foldCountsFlat[g * 4 + slot]       = features[f].Folds;
                firstFoldIndicesFlat[g * 4 + slot]  = features[f].FirstFoldIndex;
            }
        }
    }

    auto foldCountsArr     = mx::array(reinterpret_cast<const int32_t*>(foldCountsFlat.data()),
                                       {static_cast<int>(numGroups * 4)}, mx::uint32);
    auto firstFoldArr      = mx::array(reinterpret_cast<const int32_t*>(firstFoldIndicesFlat.data()),
                                       {static_cast<int>(numGroups * 4)}, mx::uint32);
    auto featureColsArr    = mx::array(reinterpret_cast<const int32_t*>(featureColumnIndicesVec.data()),
                                       {static_cast<int>(numGroups)}, mx::uint32);

    auto flatCompressed    = mx::reshape(compressedData, {-1});
    auto lineSizeArr       = mx::array(static_cast<uint32_t>(numUi32PerDoc), mx::uint32);
    auto maxBlocksArr      = mx::array(static_cast<uint32_t>(maxBlocksPerPart), mx::uint32);
    auto totalBinsArr      = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
    auto numStatsArr       = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
    auto totalDocsArr      = mx::array(static_cast<uint32_t>(numDocs), mx::uint32);
    auto numGroupsArr      = mx::array(static_cast<uint32_t>(numGroups), mx::uint32);

    // Reshape stats to [numStats, numDocs] flat
    // stats is already [numStats * numDocs] flat or we pass it as such
    mx::Shape histShape = {
        static_cast<int>(numPartitions * numStats * totalBinFeatures)
    };

    auto kernel = mx::fast::metal_kernel(
        "histogram_one_byte_features",
        {"compressedIndex", "stats", "docIndices",
         "partOffsets", "partSizes",
         "featureColumnIndices", "lineSize", "maxBlocksPerPart", "numGroups",
         "foldCountsFlat", "firstFoldIndicesFlat",
         "totalBinFeatures", "numStats", "totalNumDocs"},
        {"histogram"},
        KernelSources::kHistOneByteSource,
        KernelSources::kHistHeader,
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/true
    );

    auto grid = std::make_tuple(
        static_cast<int>(256 * maxBlocksPerPart * numGroups),
        static_cast<int>(numPartitions),
        static_cast<int>(numStats)
    );
    auto tg = std::make_tuple(256, 1, 1);

    auto results = kernel(
        {flatCompressed, stats, docIndices,
         partOffsets, partSizes,
         featureColsArr, lineSizeArr, maxBlocksArr, numGroupsArr,
         foldCountsArr, firstFoldArr,
         totalBinsArr, numStatsArr, totalDocsArr},
        {histShape}, {mx::float32},
        grid, tg,
        {}, 0.0f, false, mx::Device::gpu
    );

    return results[0];
}

// ============================================================================
// Suffix-sum + split scoring (mirrors score_calcer.cpp FindBestSplitGPU)
// ============================================================================

TBestSplitProperties FindBestSplitGPU(
    const mx::array& histogram,       // [numPartitions * numStats * totalBinFeatures]
    const mx::array& partGradSums,    // [numPartitions]
    const mx::array& partHessSums,    // [numPartitions]
    const std::vector<TCFeature>& features,
    ui32 numPartitions,
    ui32 totalBinFeatures,
    ui32 numStats,
    float l2RegLambda,
    ui32 approxDim = 1
) {
    const ui32 numFeatures = static_cast<ui32>(features.size());
    if (totalBinFeatures == 0 || numPartitions == 0) return {};

    // Build feature metadata arrays
    std::vector<uint32_t> firstFoldVec(numFeatures), foldsVec(numFeatures), isOneHotVec(numFeatures);
    std::vector<uint32_t> binToFeatureVec(totalBinFeatures);
    for (ui32 f = 0; f < numFeatures; ++f) {
        firstFoldVec[f] = features[f].FirstFoldIndex;
        foldsVec[f]     = features[f].Folds;
        isOneHotVec[f]  = features[f].OneHotFeature ? 1u : 0u;
        for (ui32 b = features[f].FirstFoldIndex; b < features[f].FirstFoldIndex + features[f].Folds; ++b) {
            binToFeatureVec[b] = f;
        }
    }

    auto firstFoldArr  = mx::array(reinterpret_cast<const int32_t*>(firstFoldVec.data()),
                                   {static_cast<int>(numFeatures)}, mx::uint32);
    auto foldsArr      = mx::array(reinterpret_cast<const int32_t*>(foldsVec.data()),
                                   {static_cast<int>(numFeatures)}, mx::uint32);
    auto isOneHotArr   = mx::array(reinterpret_cast<const int32_t*>(isOneHotVec.data()),
                                   {static_cast<int>(numFeatures)}, mx::uint32);
    auto binToFeatureArr = mx::array(reinterpret_cast<const int32_t*>(binToFeatureVec.data()),
                                    {static_cast<int>(totalBinFeatures)}, mx::uint32);

    auto numFeatArr    = mx::array(static_cast<uint32_t>(numFeatures),     mx::uint32);
    auto totalBinsArr  = mx::array(static_cast<uint32_t>(totalBinFeatures),mx::uint32);
    auto numStatsArr   = mx::array(static_cast<uint32_t>(numStats),        mx::uint32);
    auto numPartsArr   = mx::array(static_cast<uint32_t>(numPartitions),   mx::uint32);
    auto approxDimArr  = mx::array(static_cast<uint32_t>(approxDim),       mx::uint32);
    auto l2Arr         = mx::array(l2RegLambda, mx::float32);

    // Phase A: Suffix-sum transform
    auto suffixKernel = mx::fast::metal_kernel(
        "suffix_sum_histogram",
        {"histogram", "featureFirstFold", "featureFolds", "featureIsOneHot",
         "numFeatures", "totalBinFeatures", "numStats"},
        {"histogram_out"},
        KernelSources::kSuffixSumSource,
        KernelSources::kScoreHeader,
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false
    );

    auto suffixGrid = std::make_tuple(
        static_cast<int>(numFeatures),
        static_cast<int>(approxDim * numPartitions),
        static_cast<int>(numStats)
    );
    // BUG-001 FIX: 256 threads — one per bin (max 255 bins + 1 guard lane).
    // Changed from (32,1,1) to match the new Hillis-Steele 256-thread scan kernel.
    // init_value=0.0f: zero-initialise histogram_out so that one-hot feature bins
    // (not written by the kernel) and the skipped last ordinal bin read as 0.
    auto suffixTG = std::make_tuple(256, 1, 1);

    auto suffixResult = suffixKernel(
        {histogram, firstFoldArr, foldsArr, isOneHotArr,
         numFeatArr, totalBinsArr, numStatsArr},
        {histogram.shape()}, {mx::float32},
        suffixGrid, suffixTG,
        {}, 0.0f, false, mx::Device::gpu
    );
    auto transformedHist = suffixResult[0];

    // Phase B: Score splits with lookup table
    const ui32 numBlocks = (totalBinFeatures + 255) / 256;

    auto scoreKernel = mx::fast::metal_kernel(
        "score_splits_lookup",
        {"histogram", "partTotalSum", "partTotalWeight",
         "featureFirstFold", "featureFolds", "featureIsOneHot", "binToFeature",
         "numFeatures", "totalBinFeatures", "numStats", "l2RegLambda",
         "numPartitions", "approxDim"},
        {"bestScores", "bestFeatureIds", "bestBinIds"},
        KernelSources::kScoreSplitsLookupSource,
        KernelSources::kScoreHeader,
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false
    );

    auto scoreGrid = std::make_tuple(static_cast<int>(256 * numBlocks), 1, 1);
    auto scoreTG   = std::make_tuple(256, 1, 1);

    auto scoreResult = scoreKernel(
        {transformedHist, partGradSums, partHessSums,
         firstFoldArr, foldsArr, isOneHotArr, binToFeatureArr,
         numFeatArr, totalBinsArr, numStatsArr, l2Arr,
         numPartsArr, approxDimArr},
        {{static_cast<int>(numBlocks)}, {static_cast<int>(numBlocks)}, {static_cast<int>(numBlocks)}},
        {mx::float32, mx::uint32, mx::uint32},
        scoreGrid, scoreTG,
        {}, std::nullopt, false, mx::Device::gpu
    );

    mx::eval({scoreResult[0], scoreResult[1], scoreResult[2]});

    const float*    scores  = scoreResult[0].data<float>();
    const uint32_t* featIds = scoreResult[1].data<uint32_t>();
    const uint32_t* binIds  = scoreResult[2].data<uint32_t>();

    TBestSplitProperties best;
    float bestGain = -std::numeric_limits<float>::infinity();
    for (ui32 i = 0; i < numBlocks; ++i) {
        if (scores[i] > bestGain) {
            bestGain = scores[i];
            best.FeatureId = featIds[i];
            best.BinId     = binIds[i];
            best.Gain      = bestGain;
            best.Score     = -bestGain;
        }
    }
    return best;
}

// ============================================================================
// Leaf accumulation (mirrors leaf_estimator.cpp ComputeLeafSumsGPU)
// ============================================================================

void ComputeLeafSumsGPU(
    const mx::array& gradients,   // [approxDim * numDocs]
    const mx::array& hessians,    // [approxDim * numDocs]
    const mx::array& partitions,  // [numDocs] uint32
    ui32 numDocs, ui32 numLeaves, ui32 approxDim,
    mx::array& gradSumsOut, mx::array& hessSumsOut
) {
    gradSumsOut = mx::zeros({static_cast<int>(approxDim * numLeaves)}, mx::float32);
    hessSumsOut = mx::zeros({static_cast<int>(approxDim * numLeaves)}, mx::float32);

    auto numDocsArr    = mx::array(static_cast<uint32_t>(numDocs),    mx::uint32);
    auto numLeavesArr  = mx::array(static_cast<uint32_t>(numLeaves),  mx::uint32);
    auto approxDimArr  = mx::array(static_cast<uint32_t>(approxDim),  mx::uint32);

    auto leafKernel = mx::fast::metal_kernel(
        "leaf_accum",
        {"gradients", "hessians", "partitions", "numDocs", "numLeaves", "approxDim"},
        {"gradSums", "hessSums"},
        KernelSources::kLeafAccumSource,
        KernelSources::kLeafAccumHeader,
        /*ensure_row_contiguous=*/true,
        // BUG-001 FIX: atomic_outputs=false — the new single-threadgroup kernel
        // writes directly (non-atomically) since no other threadgroup touches the
        // same output slot.  The init_value=0.0f zeros the output before the kernel.
        /*atomic_outputs=*/false
    );

    // BUG-001 FIX: Single-threadgroup dispatch (grid = LEAF_BLOCK_SIZE).
    // The kernel now iterates over all numDocs internally with stride LEAF_BLOCK_SIZE.
    // This eliminates cross-threadgroup atomic_fetch_add races on the leaf sum slots.
    // Performance: serialised over numDocs within one threadgroup. For typical
    // numDocs <= 10k this is fast enough; for production workloads leaf estimation
    // is not the critical path (histogram is ~50x larger).
    auto grid = std::make_tuple(256, 1, 1);
    auto tg   = std::make_tuple(256, 1, 1);

    auto results = leafKernel(
        {gradients, hessians, partitions, numDocsArr, numLeavesArr, approxDimArr},
        {{static_cast<int>(approxDim * numLeaves)}, {static_cast<int>(approxDim * numLeaves)}},
        {mx::float32, mx::float32},
        grid, tg,
        {}, 0.0f, false, mx::Device::gpu
    );
    gradSumsOut = results[0];
    hessSumsOut = results[1];
}

// ============================================================================
// Compute leaf values: Newton step  value[l] = -gradSum[l] / (hessSum[l] + l2) * lr
// ============================================================================

mx::array ComputeLeafValues(
    const mx::array& gradSums, const mx::array& hessSums,
    float l2RegLambda, float learningRate
) {
    auto lambda = mx::array(l2RegLambda, mx::float32);
    auto lr     = mx::array(learningRate, mx::float32);
    return mx::multiply(
        mx::negative(mx::divide(gradSums, mx::add(hessSums, lambda))),
        lr
    );
}

// ============================================================================
// Update partitions after a split
// Each doc goes right (leaf |= (1 << depth)) if bin > threshold.
// ============================================================================

void ApplySplitToPartitions(
    mx::array& partitions,       // [numDocs] uint32 — updated in place
    const mx::array& compressedData, // [numDocs, numUi32PerDoc] uint32
    const TCFeature& splitFeature,
    ui32 binThreshold,
    ui32 depth,
    ui32 numDocs,
    ui32 numUi32PerDoc
) {
    // Read to CPU for the split application (mirrors csv_train's approach)
    mx::eval({partitions, compressedData});
    const uint32_t* partsPtr  = partitions.data<uint32_t>();
    const uint32_t* dataPtr   = compressedData.data<uint32_t>();

    std::vector<uint32_t> newParts(numDocs);
    const ui32 shift = splitFeature.Shift;
    const ui32 mask  = splitFeature.Mask;
    const ui32 col   = static_cast<ui32>(splitFeature.Offset);

    for (ui32 d = 0; d < numDocs; ++d) {
        uint32_t featureVal = (dataPtr[d * numUi32PerDoc + col] >> shift) & mask;
        // Right child if value > threshold (CatBoost ordinal convention)
        bool goRight = (featureVal > binThreshold + 1);
        newParts[d] = partsPtr[d] | (goRight ? (1u << depth) : 0u);
    }
    partitions = mx::array(reinterpret_cast<const int32_t*>(newParts.data()),
                           {static_cast<int>(numDocs)}, mx::uint32);
}

// ============================================================================
// Compute RMSE loss (binary: uses sigmoid; multiclass: uses raw index accuracy)
// ============================================================================

float ComputeLoss(
    const mx::array& cursor,   // [numDocs] for regression/binary, [K, numDocs] for multiclass
    const mx::array& targets,  // [numDocs]
    ui32 numClasses
) {
    if (numClasses == 1) {
        // RMSE
        auto diff   = mx::subtract(cursor, targets);
        auto sq     = mx::multiply(diff, diff);
        auto rmse   = mx::sqrt(mx::mean(sq));
        mx::eval(rmse);
        return rmse.item<float>();
    } else if (numClasses == 2) {
        // Logloss
        auto sig    = mx::sigmoid(cursor);
        auto eps    = mx::array(1e-15f);
        auto logSig = mx::log(mx::add(sig, eps));
        auto log1m  = mx::log(mx::add(mx::subtract(mx::array(1.0f), sig), eps));
        auto loss   = mx::negative(mx::mean(mx::add(
            mx::multiply(targets, logSig),
            mx::multiply(mx::subtract(mx::array(1.0f), targets), log1m)
        )));
        mx::eval(loss);
        return loss.item<float>();
    } else {
        // Multiclass: cross-entropy with implicit K-th class
        // cursor: [K, numDocs], K = numClasses - 1
        const int K       = static_cast<int>(numClasses - 1);
        const int numDocs = cursor.shape(-1);

        auto maxCursor   = mx::max(cursor, 0);
        maxCursor        = mx::maximum(maxCursor, mx::array(0.0f));
        auto expCursor   = mx::exp(mx::subtract(cursor, mx::reshape(maxCursor, {1, numDocs})));
        auto expImplicit = mx::exp(mx::negative(maxCursor));
        auto sumExp      = mx::add(mx::sum(expCursor, 0), expImplicit);
        auto probs       = mx::divide(expCursor, mx::reshape(sumExp, {1, numDocs}));

        auto targetInt   = mx::astype(targets, mx::int32);
        auto probTarget  = mx::zeros({numDocs}, mx::float32);
        for (int k = 0; k < K; ++k) {
            auto isClass = mx::astype(mx::equal(targetInt, mx::array(k)), mx::float32);
            auto probK   = mx::reshape(mx::slice(probs, {k, 0}, {k+1, numDocs}), {numDocs});
            probTarget   = mx::add(probTarget, mx::multiply(isClass, probK));
        }
        auto isLastClass  = mx::astype(mx::equal(targetInt, mx::array(K)), mx::float32);
        auto implicitProb = mx::subtract(mx::array(1.0f), mx::sum(probs, 0));
        probTarget        = mx::add(probTarget, mx::multiply(isLastClass, implicitProb));

        auto loss = mx::negative(mx::mean(mx::log(mx::add(probTarget, mx::array(1e-15f)))));
        mx::eval(loss);
        return loss.item<float>();
    }
}

// ============================================================================
// Single boosting iteration
//   Returns wall time in microseconds (excluding GPU graph construction overhead)
// ============================================================================

long long RunIteration(
    mx::array& cursor,           // updated in place
    mx::array& partitions,       // reset each iter
    const mx::array& compressedData,
    const mx::array& targets,
    const std::vector<TCFeature>& features,
    ui32 numDocs,
    ui32 numUi32PerDoc,
    ui32 numPartitions,          // = 2^depth
    ui32 totalBinFeatures,
    ui32 numStats,
    ui32 approxDim,
    ui32 numClasses,
    ui32 maxDepth,
    float l2RegLambda,
    float learningRate,
    ui32 maxBlocksPerPart
) {
    auto wallStart = std::chrono::steady_clock::now();

    // --- Compute gradients and hessians ---
    mx::array gradients = mx::zeros({1}, mx::float32);
    mx::array hessians  = mx::zeros({1}, mx::float32);
    if (numClasses == 1) {
        // RMSE: grad = cursor - target, hess = 1
        gradients = mx::subtract(cursor, targets);
        hessians  = mx::ones({static_cast<int>(numDocs)}, mx::float32);
    } else if (numClasses == 2) {
        // Logloss
        auto sig  = mx::sigmoid(cursor);
        gradients = mx::subtract(sig, targets);
        hessians  = mx::multiply(sig, mx::subtract(mx::array(1.0f), sig));
        hessians  = mx::maximum(hessians, mx::array(1e-16f));
    } else {
        // Multiclass: cursor [K, numDocs]
        const int K    = static_cast<int>(approxDim);
        const int nDocs= static_cast<int>(numDocs);
        auto maxC      = mx::max(cursor, 0);
        maxC           = mx::maximum(maxC, mx::array(0.0f));
        auto expC      = mx::exp(mx::subtract(cursor, mx::reshape(maxC, {1, nDocs})));
        auto expImpl   = mx::exp(mx::negative(maxC));
        auto sumExp    = mx::add(mx::sum(expC, 0), expImpl);
        auto probs     = mx::divide(expC, mx::reshape(sumExp, {1, nDocs}));
        auto targInt   = mx::astype(targets, mx::uint32);
        auto oneHot    = mx::zeros({K, nDocs}, mx::float32);
        for (int k = 0; k < K; ++k) {
            auto isClass = mx::astype(mx::equal(targInt, mx::array(static_cast<uint32_t>(k))), mx::float32);
            oneHot = mx::where(
                mx::equal(mx::reshape(mx::arange(K), {K, 1}), mx::array(k)),
                mx::reshape(isClass, {1, nDocs}),
                oneHot
            );
        }
        gradients = mx::subtract(probs, oneHot);
        hessians  = mx::multiply(probs, mx::subtract(mx::array(1.0f), probs));
        hessians  = mx::maximum(hessians, mx::array(1e-16f));
    }
    mx::eval({gradients, hessians});

    // --- Reset partitions to all-zero (single partition at start) ---
    partitions = mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
    mx::eval(partitions);

    // Layout the gradients/hessians as flat [approxDim * numDocs] for the leaf kernel
    // (gradients for approxDim==1 are already [numDocs])
    mx::array flatGrad = (approxDim == 1)
        ? mx::reshape(gradients, {static_cast<int>(numDocs)})
        : mx::reshape(gradients, {static_cast<int>(approxDim * numDocs)});
    mx::array flatHess = (approxDim == 1)
        ? mx::reshape(hessians, {static_cast<int>(numDocs)})
        : mx::reshape(hessians, {static_cast<int>(approxDim * numDocs)});

    // Stats for histogram kernel: [numStats * numDocs] flat
    // For GBDT we pass gradients as stat-0; hessians as stat-1.
    // For approxDim > 1 we process dim 0 only in the histogram (sufficient for benchmark).
    mx::array statArr = mx::zeros({1}, mx::float32);
    if (approxDim == 1) {
        statArr = mx::concatenate({
            mx::reshape(gradients, {1, static_cast<int>(numDocs)}),
            mx::reshape(hessians,  {1, static_cast<int>(numDocs)})
        }, 0);
        statArr = mx::reshape(statArr, {static_cast<int>(2 * numDocs)});
    } else {
        // Use dim 0 gradients/hessians for histogram scoring (simplified benchmark)
        auto grad0 = mx::slice(gradients, {0, 0}, {1, static_cast<int>(numDocs)});
        auto hess0 = mx::slice(hessians,  {0, 0}, {1, static_cast<int>(numDocs)});
        statArr = mx::concatenate({
            mx::reshape(grad0, {1, static_cast<int>(numDocs)}),
            mx::reshape(hess0, {1, static_cast<int>(numDocs)})
        }, 0);
        statArr = mx::reshape(statArr, {static_cast<int>(2 * numDocs)});
    }
    mx::eval(statArr);

    // --- Greedy tree structure search: one split per depth level ---
    for (ui32 depth = 0; depth < maxDepth; ++depth) {
        const ui32 numActiveParts = 1u << depth;

        // Compute partition layout
        auto layout = ComputePartitionLayout(partitions, numDocs, numActiveParts);

        // Dispatch histogram
        auto histogram = DispatchHistogram(
            compressedData, statArr,
            layout.DocIndices, layout.PartOffsets, layout.PartSizes,
            features, numUi32PerDoc, numActiveParts,
            totalBinFeatures, numStats, numDocs, maxBlocksPerPart
        );
        mx::eval(histogram);

        // Compute partition totals (grad sum, hess sum) per partition
        // Use leaf accum kernel with current gradients
        mx::array gradSumsGPU = mx::zeros({1}, mx::float32);
        mx::array hessSumsGPU = mx::zeros({1}, mx::float32);
        ComputeLeafSumsGPU(flatGrad, flatHess, partitions,
                           numDocs, numActiveParts, 1 /*approxDim for scoring*/,
                           gradSumsGPU, hessSumsGPU);
        mx::eval({gradSumsGPU, hessSumsGPU});

        // Find best split
        auto best = FindBestSplitGPU(
            histogram, gradSumsGPU, hessSumsGPU,
            features, numActiveParts, totalBinFeatures, numStats,
            l2RegLambda
        );

        if (!best.Defined()) break;

        // Apply split to partitions
        const TCFeature& splitFeat = features[best.FeatureId];
        ApplySplitToPartitions(
            partitions, compressedData, splitFeat,
            best.BinId, depth, numDocs, numUi32PerDoc
        );
    }

    // --- Estimate leaf values and apply tree ---
    const ui32 numLeaves = 1u << maxDepth;
    mx::array gradSumsGPU = mx::zeros({1}, mx::float32);
    mx::array hessSumsGPU = mx::zeros({1}, mx::float32);
    ComputeLeafSumsGPU(
        flatGrad, flatHess, partitions,
        numDocs, numLeaves, approxDim,
        gradSumsGPU, hessSumsGPU
    );
    mx::eval({gradSumsGPU, hessSumsGPU});

    if (approxDim == 1) {
        auto leafVals = ComputeLeafValues(gradSumsGPU, hessSumsGPU, l2RegLambda, learningRate);
        // Apply: cursor[d] += leafVals[partitions[d]]
        auto gathered = mx::take(leafVals, partitions, 0);
        cursor = mx::add(cursor, gathered);
        mx::eval(cursor);
    } else {
        // Multiclass: compute leaf values per dim, apply
        const int K = static_cast<int>(approxDim);
        const int nDocs = static_cast<int>(numDocs);
        const int nLeaves = static_cast<int>(numLeaves);
        std::vector<float> interleavedLeaves(numLeaves * approxDim, 0.0f);
        for (ui32 k = 0; k < approxDim; ++k) {
            auto dimGrad = mx::slice(gradSumsGPU,
                {static_cast<int>(k * numLeaves)}, {static_cast<int>((k+1) * numLeaves)});
            auto dimHess = mx::slice(hessSumsGPU,
                {static_cast<int>(k * numLeaves)}, {static_cast<int>((k+1) * numLeaves)});
            auto dimLeaf = ComputeLeafValues(dimGrad, dimHess, l2RegLambda, learningRate);
            mx::eval(dimLeaf);
            const float* lp = dimLeaf.data<float>();
            for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
                interleavedLeaves[leaf * approxDim + k] = lp[leaf];
            }
        }
        auto leafVals2D = mx::array(interleavedLeaves.data(),
            {nLeaves, K}, mx::float32);
        // Apply: cursor[k, d] += leafVals2D[partitions[d], k]
        // Use take along axis=0
        auto leafPerDoc = mx::take(leafVals2D, partitions, 0);  // [numDocs, K]
        leafPerDoc = mx::transpose(leafPerDoc);                  // [K, numDocs]
        cursor = mx::add(cursor, leafPerDoc);
        mx::eval(cursor);
    }

    auto wallEnd = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(wallEnd - wallStart).count();
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    TBenchConfig cfg = ParseArgs(argc, argv);

    const ui32 approxDim   = (cfg.NumClasses <= 2) ? 1u : (cfg.NumClasses - 1);
    const ui32 numStats    = 2;  // gradient + hessian
    const ui32 numLeaves   = 1u << cfg.MaxDepth;

    printf("bench_boosting — CatBoost-MLX library-path benchmark (TODO-014)\n");
    printf("================================================================\n");
    printf("rows=%u  features=%u  classes=%u  depth=%u  iters=%u  bins=%u\n",
           cfg.NumRows, cfg.NumFeatures, cfg.NumClasses,
           cfg.MaxDepth, cfg.NumIters, cfg.NumBins);
    printf("lr=%.4f  l2=%.4f  seed=%llu  approxDim=%u  onehot=%u\n",
           cfg.LearningRate, cfg.L2RegLambda, (unsigned long long)cfg.Seed, approxDim,
           std::min(cfg.NumOneHot, cfg.NumFeatures));
    printf("----------------------------------------------------------------\n");

    // Generate synthetic dataset
    printf("Generating synthetic dataset...\n");
    auto ds = GenerateSynthDataset(cfg);

    // Upload to GPU
    auto compressedData = mx::array(
        reinterpret_cast<const int32_t*>(ds.CompressedData.data()),
        {static_cast<int>(cfg.NumRows), static_cast<int>(ds.NumUi32PerDoc)},
        mx::uint32
    );
    auto targets = mx::array(ds.Targets.data(), {static_cast<int>(cfg.NumRows)}, mx::float32);
    mx::eval({compressedData, targets});

    // Initialize cursor
    mx::array cursor = mx::zeros({1}, mx::float32);
    if (approxDim == 1) {
        cursor = mx::zeros({static_cast<int>(cfg.NumRows)}, mx::float32);
    } else {
        cursor = mx::zeros({static_cast<int>(approxDim), static_cast<int>(cfg.NumRows)}, mx::float32);
    }
    mx::eval(cursor);

    mx::array partitions = mx::zeros({static_cast<int>(cfg.NumRows)}, mx::uint32);
    mx::eval(partitions);

    // BUG-001 FIX: Use maxBlocksPerPart=1 (matches production histogram.cpp default).
    // maxBlocksPerPart=4 caused multiple threadgroups to write to the same histogram
    // slot via atomic_fetch_add_explicit, producing non-deterministic float addition
    // ordering across dispatches. With maxBlocksPerPart=1, each partition-feature-group
    // combination is handled by exactly one threadgroup, eliminating cross-threadgroup
    // float-add races in the global histogram buffer.
    const ui32 maxBlocksPerPart = 1;

    printf("Dataset: %u rows x %u features (%u uint32 cols), %u total bin-features\n",
           cfg.NumRows, cfg.NumFeatures, ds.NumUi32PerDoc, ds.TotalBinFeatures);
    printf("Beginning %u boosting iterations...\n\n", cfg.NumIters);

    std::vector<long long> iterTimesUs;
    iterTimesUs.reserve(cfg.NumIters);

    for (ui32 iter = 0; iter < cfg.NumIters; ++iter) {
        long long us = RunIteration(
            cursor, partitions,
            compressedData, targets,
            ds.Features,
            cfg.NumRows, ds.NumUi32PerDoc,
            1u,                    // single partition at start of each iter
            ds.TotalBinFeatures, numStats, approxDim, cfg.NumClasses,
            cfg.MaxDepth, cfg.L2RegLambda, cfg.LearningRate,
            maxBlocksPerPart
        );
        iterTimesUs.push_back(us);

        if (iter == 0 || (iter + 1) % 10 == 0 || iter == cfg.NumIters - 1) {
            // Compute current loss
            float loss = ComputeLoss(cursor, targets, cfg.NumClasses);
            printf("  iter %4u  time=%7.1f ms  loss=%.6f\n",
                   iter, us / 1000.0f, loss);
        }
    }

    // Summary
    printf("\n================================================================\n");
    printf("Timing Summary\n");
    printf("----------------------------------------------------------------\n");

    long long iter0Us = iterTimesUs[0];
    double warmSumUs  = 0.0;
    long long warmMin = std::numeric_limits<long long>::max();
    long long warmMax = 0;

    for (ui32 i = 1; i < iterTimesUs.size(); ++i) {
        warmSumUs += iterTimesUs[i];
        warmMin = std::min(warmMin, iterTimesUs[i]);
        warmMax = std::max(warmMax, iterTimesUs[i]);
    }
    ui32 warmCount = (ui32)iterTimesUs.size() - 1;
    double warmMeanUs = warmCount > 0 ? warmSumUs / warmCount : 0.0;

    printf("  iter-0 (cold start):  %8.1f ms  [Metal kernel compile + first dispatch]\n",
           iter0Us / 1000.0f);
    if (warmCount > 0) {
        printf("  warm mean (%4u iters): %8.1f ms\n", warmCount, warmMeanUs / 1000.0f);
        printf("  warm min:             %8.1f ms\n", warmMin / 1000.0f);
        printf("  warm max:             %8.1f ms\n", warmMax / 1000.0f);
    }

    // Final loss (for regression testing)
    float finalLoss = ComputeLoss(cursor, targets, cfg.NumClasses);
    printf("\n  Final loss: %.8f\n", finalLoss);
    printf("  BENCH_FINAL_LOSS=%.8f\n", finalLoss);  // grep-friendly line

    printf("================================================================\n");

    return 0;
}
