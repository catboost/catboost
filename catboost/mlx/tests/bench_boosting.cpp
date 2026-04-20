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
//     --rows N               Number of training documents (default: 100000)
//     --features N           Number of features (default: 50)
//     --classes N            Number of classes: 1 = regression, 2 = binary, K >= 3 = multiclass (default: 1)
//     --depth D              Max tree depth (default: 6)
//     --iters N              Number of boosting iterations (default: 100)
//     --bins B               Bins per feature (default: 32; max 255)
//     --lr F                 Learning rate (default: 0.1)
//     --l2 F                 L2 regularization lambda (default: 3.0)
//     --seed N               Random seed for data synthesis (default: 42)
//     --onehot N             Mark the first N features as one-hot encoded (default: 0)
//     --per-kernel-profile   Insert mx::eval() sync points between kernel stages and print
//                            a per-kernel timing table (UPPER BOUNDS — disables kernel overlap)
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
#include <fstream>
#include <sstream>

#ifdef CATBOOST_MLX_STAGE_PROFILE
#include <filesystem>
#endif

// Sprint 23 D0: T2 sort-by-bin histogram dispatch promoted to production.
// DispatchHistogramT2 implementation lives in catboost/mlx/methods/histogram.cpp.
// It is declared below (forward declaration) to avoid including histogram.h,
// which has CatBoost-type dependencies incompatible with this standalone bench.
// The #ifdef CATBOOST_MLX_HISTOGRAM_T2 call-site guards are removed in Commit 3.

namespace mx = mlx::core;
using ui32 = uint32_t;
using ui64 = uint64_t;

namespace NCatboostMlx {
namespace KernelSources {}  // from kernel_sources.h include above

// Forward declaration of the production T2 dispatch (implemented in histogram.cpp).
// Takes pre-built fold-metadata arrays; fold construction is done by the
// bench-local BuildFoldArraysForT2() helper below.
mx::array DispatchHistogramT2(
    const mx::array& compressedData,
    const mx::array& stats,
    const mx::array& docIndices,
    const mx::array& partOffsets,
    const mx::array& partSizes,
    const mx::array& featureColIndices,
    const mx::array& foldCountsFlat,
    const mx::array& firstFoldFlat,
    ui32 lineSize,
    ui32 maxBlocksPerPart,
    ui32 numGroups,
    ui32 numPartitions,
    ui32 numStats,
    ui32 totalBinFeatures,
    ui32 totalNumDocs,
    const mx::Shape& histShape
);
}  // namespace NCatboostMlx

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
    ui32 NumOneHot        = 0;     // first N features treated as one-hot encoded
    bool StageProfile     = false; // --stage-profile: emit per-iter stage JSON
    bool PerKernelProfile = false; // --per-kernel-profile: per-dispatch timing (upper bounds)
    bool UseT2            = false; // --t2: Sprint 22 D0 probe — use T2 sort-by-bin histogram
                                   //       Requires -DCATBOOST_MLX_HISTOGRAM_T2=1 at compile time
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
        else if (strcmp(argv[i], "--onehot")              == 0 && i+1 < argc) cfg.NumOneHot      = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--stage-profile")       == 0) cfg.StageProfile      = true;
        else if (strcmp(argv[i], "--per-kernel-profile")  == 0) cfg.PerKernelProfile  = true;
        else if (strcmp(argv[i], "--t2")                  == 0) cfg.UseT2             = true;
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr,
                "Usage: bench_boosting [--rows N] [--features N] [--classes N]\n"
                "                      [--depth D] [--iters N] [--bins B]\n"
                "                      [--lr F] [--l2 F] [--seed N] [--onehot N]\n"
                "                      [--stage-profile] [--per-kernel-profile]\n"
                "                      [--t2]  (Sprint 22 D0 probe; needs -DCATBOOST_MLX_HISTOGRAM_T2=1)\n");
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
    //
    // Semantics note (S19-13): real-quantize (csv_train.cpp) stores Folds = numBorders
    // (for no-NaN features), where numBorders = cfg.NumBins - 1, and the actual bin
    // values land in [0, numBorders]. Bench's synth path mirrors that: ordinal
    // Folds = cfg.NumBins - 1, and `binDist` below draws from [0, folds]. Previously
    // bench stored Folds = cfg.NumBins, which over-reported by one vs real-quantize.
    // Alignment matters for the DEC-016 T1 MSB-sentinel envelope guard
    // (DispatchHistogram in this file + DispatchHistogramBatched in histogram.cpp).
    ui32 totalBinFeatures = 0;
    ds.Features.resize(cfg.NumFeatures);
    for (ui32 f = 0; f < cfg.NumFeatures; ++f) {
        const bool isOneHot = (f < numOneHot);
        // One-hot bin count: derive a small value (2–10) deterministically from f and seed
        // so different features get different sizes, and the result is reproducible.
        const ui32 folds = isOneHot
            ? static_cast<ui32>(2 + ((cfg.Seed + f) % 9))  // 2..10
            : (cfg.NumBins > 0u ? cfg.NumBins - 1u : 0u);

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

    ui32 maxFoldCount = 0;
    for (ui32 g = 0; g < numGroups; ++g) {
        featureColumnIndicesVec[g] = g;  // each group maps to column g
        for (ui32 slot = 0; slot < 4; ++slot) {
            ui32 f = g * 4 + slot;
            if (f < numFeatures) {
                const ui32 folds = features[f].Folds;
                foldCountsFlat[g * 4 + slot]        = folds;
                firstFoldIndicesFlat[g * 4 + slot]  = features[f].FirstFoldIndex;
                if (folds > maxFoldCount) maxFoldCount = folds;
            }
        }
    }

    // DEC-016 T1 envelope guard — see histogram.cpp for full rationale.
    if (maxFoldCount > 127u) {
        std::fprintf(stderr,
                     "FATAL: bench_boosting histogram kernel: max fold count %u "
                     "exceeds DEC-016 T1 envelope (<= 127). The MSB-sentinel "
                     "collides with bin values >= 128.\n",
                     maxFoldCount);
        std::exit(1);
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
// Sprint 23 D0 — bench-local T2 call helper (Commit 2).
//
// Bridges the bench's local TCFeature vector to the production DispatchHistogramT2
// signature (which takes pre-built mx::array fold metadata).  This wrapper replaces
// the inline scratch implementation removed from bench_boosting.cpp in Commit 2.
//
// The #ifdef CATBOOST_MLX_HISTOGRAM_T2 guards at the call sites remain here;
// they are removed globally in Commit 3 when the flag is retired.
// ============================================================================

#ifdef CATBOOST_MLX_HISTOGRAM_T2

// Thin wrapper: builds fold arrays from features, then calls the production dispatch.
static mx::array CallDispatchHistogramT2(
    const mx::array& compressedData,
    const mx::array& stats,
    const mx::array& docIndices,
    const mx::array& partOffsets,
    const mx::array& partSizes,
    const std::vector<TCFeature>& features,
    ui32 numUi32PerDoc,
    ui32 numActiveParts,
    ui32 totalBinFeatures,
    ui32 numStats,
    ui32 numDocs,
    ui32 maxBlocksPerPart
) {
    const ui32 numFeatures = static_cast<ui32>(features.size());
    const ui32 numGroups   = (numFeatures + 3) / 4;

    std::vector<uint32_t> foldCountsFlatVec(numGroups * 4, 0u);
    std::vector<uint32_t> firstFoldIndicesFlatVec(numGroups * 4, 0u);
    std::vector<uint32_t> featureColIndicesVec(numGroups, 0u);
    ui32 maxFoldCount = 0;
    for (ui32 g = 0; g < numGroups; ++g) {
        featureColIndicesVec[g] = g;
        for (ui32 slot = 0; slot < 4; ++slot) {
            ui32 f = g * 4 + slot;
            if (f < numFeatures) {
                const ui32 folds = features[f].Folds;
                foldCountsFlatVec[g * 4 + slot]      = folds;
                firstFoldIndicesFlatVec[g * 4 + slot] = features[f].FirstFoldIndex;
                if (folds > maxFoldCount) maxFoldCount = folds;
            }
        }
    }

    if (maxFoldCount > 127u) {
        std::fprintf(stderr,
            "FATAL: CallDispatchHistogramT2: max fold count %u exceeds DEC-016 envelope (<=127).\n",
            maxFoldCount);
        std::exit(1);
    }

    auto foldCountsArr  = mx::array(reinterpret_cast<const int32_t*>(foldCountsFlatVec.data()),
                                    {static_cast<int>(numGroups * 4)}, mx::uint32);
    auto firstFoldArr   = mx::array(reinterpret_cast<const int32_t*>(firstFoldIndicesFlatVec.data()),
                                    {static_cast<int>(numGroups * 4)}, mx::uint32);
    auto featureColsArr = mx::array(reinterpret_cast<const int32_t*>(featureColIndicesVec.data()),
                                    {static_cast<int>(numGroups)}, mx::uint32);

    mx::Shape histShape = {static_cast<int>(numActiveParts * numStats * totalBinFeatures)};

    return NCatboostMlx::DispatchHistogramT2(
        compressedData, stats,
        docIndices, partOffsets, partSizes,
        featureColsArr, foldCountsArr, firstFoldArr,
        numUi32PerDoc, maxBlocksPerPart,
        numGroups, numActiveParts, numStats,
        totalBinFeatures, numDocs,
        histShape
    );
}

#endif  // CATBOOST_MLX_HISTOGRAM_T2

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
    ui32 approxDim = 1,
    // Per-kernel profile mode: insert eval() sync between suffix and score phases,
    // capture sub-timings. When false (default), code path is identical to before.
    bool perKernelProfile = false,
    double* suffixMsOut   = nullptr,  // out: suffix_sum wall time (only when perKernelProfile)
    double* scoreMsOut    = nullptr   // out: score_splits wall time (only when perKernelProfile)
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

    // Per-kernel profile: capture time before suffix dispatch
    using hrc = std::chrono::high_resolution_clock;
    auto suf_t0 = perKernelProfile ? hrc::now() : hrc::time_point{};

    auto suffixResult = suffixKernel(
        {histogram, firstFoldArr, foldsArr, isOneHotArr,
         numFeatArr, totalBinsArr, numStatsArr},
        {histogram.shape()}, {mx::float32},
        suffixGrid, suffixTG,
        {}, 0.0f, false, mx::Device::gpu
    );
    auto transformedHist = suffixResult[0];

    // Per-kernel profile: force suffix sync, record time, start score timer
    if (perKernelProfile) {
        mx::eval(transformedHist);
        auto suf_t1 = hrc::now();
        if (suffixMsOut)
            *suffixMsOut = std::chrono::duration<double, std::milli>(suf_t1 - suf_t0).count();
    }
    auto sc_t0 = perKernelProfile ? hrc::now() : hrc::time_point{};

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

    // Per-kernel profile: record score phase wall time (GPU dispatch + CPU reduction)
    if (perKernelProfile && scoreMsOut) {
        auto sc_t1 = hrc::now();
        *scoreMsOut = std::chrono::duration<double, std::milli>(sc_t1 - sc_t0).count();
    }

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

// Single-pass: numLeaves <= 64 (depth <= 6)
static void ComputeLeafSumsGPUSinglePass(
    const mx::array& gradients, const mx::array& hessians,
    const mx::array& partitions,
    ui32 numDocs, ui32 numLeaves, ui32 approxDim,
    mx::array& gradSumsOut, mx::array& hessSumsOut
) {
    auto numDocsArr   = mx::array(static_cast<uint32_t>(numDocs),   mx::uint32);
    auto numLeavesArr = mx::array(static_cast<uint32_t>(numLeaves), mx::uint32);
    auto approxDimArr = mx::array(static_cast<uint32_t>(approxDim), mx::uint32);

    auto leafKernel = mx::fast::metal_kernel(
        "leaf_accum",
        {"gradients", "hessians", "partitions", "numDocs", "numLeaves", "approxDim"},
        {"gradSums", "hessSums"},
        KernelSources::kLeafAccumSource,
        KernelSources::kLeafAccumHeader,
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false
    );

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

// Multi-pass: numLeaves > 64 (depth 7-10)
// Dispatches ceil(numLeaves/64) passes of the chunked kernel; each pass processes
// a 64-leaf window. The private array stays 5 KB regardless of total numLeaves.
static void ComputeLeafSumsGPUMultiPass(
    const mx::array& gradients, const mx::array& hessians,
    const mx::array& partitions,
    ui32 numDocs, ui32 numLeaves, ui32 approxDim,
    mx::array& gradSumsOut, mx::array& hessSumsOut
) {
    constexpr ui32 kChunkSize = 64;

    auto chunkedKernel = mx::fast::metal_kernel(
        "leaf_accum_chunked",
        {"gradients", "hessians", "partitions",
         "numDocs", "chunkBase", "chunkSize", "approxDim"},
        {"gradSums", "hessSums"},
        KernelSources::kLeafAccumChunkedSource,
        KernelSources::kLeafAccumHeader,
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false
    );

    auto grid = std::make_tuple(256, 1, 1);
    auto tg   = std::make_tuple(256, 1, 1);

    auto numDocsArr   = mx::array(static_cast<uint32_t>(numDocs),   mx::uint32);
    auto approxDimArr = mx::array(static_cast<uint32_t>(approxDim), mx::uint32);

    std::vector<float> gradBuf(approxDim * numLeaves, 0.0f);
    std::vector<float> hessBuf(approxDim * numLeaves, 0.0f);

    for (ui32 chunkBase = 0; chunkBase < numLeaves; chunkBase += kChunkSize) {
        const ui32 chunkSize = std::min(kChunkSize, numLeaves - chunkBase);

        auto chunkBaseArr = mx::array(static_cast<uint32_t>(chunkBase), mx::uint32);
        auto chunkSizeArr = mx::array(static_cast<uint32_t>(chunkSize), mx::uint32);

        auto results = chunkedKernel(
            {gradients, hessians, partitions,
             numDocsArr, chunkBaseArr, chunkSizeArr, approxDimArr},
            {{static_cast<int>(approxDim * chunkSize)},
             {static_cast<int>(approxDim * chunkSize)}},
            {mx::float32, mx::float32},
            grid, tg,
            {}, 0.0f, false, mx::Device::gpu
        );

        mx::eval({results[0], results[1]});

        const float* gp = results[0].data<float>();
        const float* hp = results[1].data<float>();

        for (ui32 k = 0; k < approxDim; ++k) {
            for (ui32 li = 0; li < chunkSize; ++li) {
                gradBuf[k * numLeaves + chunkBase + li] = gp[k * chunkSize + li];
                hessBuf[k * numLeaves + chunkBase + li] = hp[k * chunkSize + li];
            }
        }
    }

    gradSumsOut = mx::array(gradBuf.data(),
        {static_cast<int>(approxDim * numLeaves)}, mx::float32);
    hessSumsOut = mx::array(hessBuf.data(),
        {static_cast<int>(approxDim * numLeaves)}, mx::float32);
}

void ComputeLeafSumsGPU(
    const mx::array& gradients,   // [approxDim * numDocs]
    const mx::array& hessians,    // [approxDim * numDocs]
    const mx::array& partitions,  // [numDocs] uint32
    ui32 numDocs, ui32 numLeaves, ui32 approxDim,
    mx::array& gradSumsOut, mx::array& hessSumsOut
) {
    if (numLeaves <= 64) {
        ComputeLeafSumsGPUSinglePass(
            gradients, hessians, partitions,
            numDocs, numLeaves, approxDim,
            gradSumsOut, hessSumsOut);
    } else {
        ComputeLeafSumsGPUMultiPass(
            gradients, hessians, partitions,
            numDocs, numLeaves, approxDim,
            gradSumsOut, hessSumsOut);
    }
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
        bool goRight = (featureVal > binThreshold);
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
// Stage profile record (bench_boosting coarse stages — 3 buckets)
// ============================================================================
//
// bench_boosting uses its own standalone pipeline, not RunBoosting/structure_searcher.
// Fine-grained depth-level stages 3/4/5 are only available via the real RunBoosting
// path (csv_train compiled with -DCATBOOST_MLX_STAGE_PROFILE).
// This provides a coarser 3-bucket view sufficient for high-level attribution.

struct TBenchStageRecord {
    int    Iter            = 0;
    double DerivativesMs   = 0.0;  // Stage 1+2: gradients, hessians, partitions
    double TreeSearchMs    = 0.0;  // Stages 3+4+5: layout + histogram + scoring
    double LeafEstimMs     = 0.0;  // Stages 6+7+8: leaf sums + Newton step + apply
    double IterTotalMs     = 0.0;
};

static bool WriteBenchStageJson(
    const std::vector<TBenchStageRecord>& records,
    const std::string& path,
    const std::string& meta
) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[bench_boosting] Cannot open profile output: %s\n", path.c_str());
        return false;
    }
    f << "{\n  \"meta\": " << meta << ",\n";
    f << "  \"note\": \"bench_boosting coarse 3-stage profile; compile with -DCATBOOST_MLX_STAGE_PROFILE for fine-grained 9-stage+depth breakdown\",\n";
    f << "  \"stage_names\": [\"derivatives_ms\", \"tree_search_ms\", \"leaf_estimation_ms\"],\n";
    f << "  \"iterations\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        f << "    {\"iter\": " << r.Iter
          << ", \"derivatives_ms\": " << r.DerivativesMs
          << ", \"tree_search_ms\": " << r.TreeSearchMs
          << ", \"leaf_estimation_ms\": " << r.LeafEstimMs
          << ", \"iter_total_ms\": " << r.IterTotalMs
          << "}";
        if (i + 1 < records.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n}\n";
    f.flush();
    return f.good();
}

// ============================================================================
// Per-kernel timing collection (--per-kernel-profile)
//
// One entry per boosting iteration (including iter-0 cold start).
// Reporting skips index-0 (matches warm_mean convention).
//
// NOTE: mx::eval() sync points between stages suppress MLX kernel overlap.
// All values are UPPER BOUNDS on non-sync production timing.
// ============================================================================

struct TPerKernelTimings {
    std::vector<double> DerivativesMs;   // grad/hess + partitions + statArr assembly
    std::vector<double> TreeSupportMs;   // ComputePartitionLayout + scoring leaf accum × depth
    std::vector<double> HistogramMs;     // DispatchHistogram × depth
    std::vector<double> SuffixSumMs;     // suffix_sum_histogram × depth
    std::vector<double> SplitScoreMs;    // score_splits_lookup + CPU reduction × depth
    std::vector<double> LeafMs;          // final ComputeLeafSumsGPU + ComputeLeafValues + cursor
    std::vector<double> IterTotalMs;     // wall-clock for the full iteration (per-kernel path)
};

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
    ui32 maxBlocksPerPart,
    TBenchStageRecord*   stageOut = nullptr,  // optional per-iter stage record
    TPerKernelTimings*   pkOut    = nullptr,  // optional per-kernel timing record
    bool                 useT2   = false      // Sprint 22 D0: use T2 sort-by-bin histogram
) {
    auto wallStart = std::chrono::steady_clock::now();

    // --- Compute gradients and hessians ---
    auto t0 = std::chrono::steady_clock::now();
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
    auto t1 = std::chrono::steady_clock::now();  // end of derivatives/init stage

    // Per-kernel: record derivatives bucket
    using hrc = std::chrono::high_resolution_clock;
    using ms_d = std::chrono::duration<double, std::milli>;
    if (pkOut) {
        pkOut->DerivativesMs.push_back(ms_d(t1 - t0).count());
    }

    // Per-kernel: depth-loop accumulators (reset each iteration)
    double pk_treeSupportAccum = 0.0;
    double pk_histAccum        = 0.0;
    double pk_suffixAccum      = 0.0;
    double pk_scoreAccum       = 0.0;

    // --- Greedy tree structure search: one split per depth level ---
    for (ui32 depth = 0; depth < maxDepth; ++depth) {
        const ui32 numActiveParts = 1u << depth;

        // Compute partition layout
        if (pkOut) {
            // Force-eval the layout arrays to isolate partition-layout cost
            auto layout_t0 = hrc::now();
            auto layout = ComputePartitionLayout(partitions, numDocs, numActiveParts);
            mx::eval({layout.DocIndices, layout.PartOffsets, layout.PartSizes});
            auto layout_t1 = hrc::now();

            // Scoring leaf accum (uses already-eval'd layout via partitions)
            mx::array gradSumsGPU = mx::zeros({1}, mx::float32);
            mx::array hessSumsGPU = mx::zeros({1}, mx::float32);
            auto leafscr_t0 = hrc::now();
            ComputeLeafSumsGPU(flatGrad, flatHess, partitions,
                               numDocs, numActiveParts, 1 /*approxDim for scoring*/,
                               gradSumsGPU, hessSumsGPU);
            mx::eval({gradSumsGPU, hessSumsGPU});
            auto leafscr_t1 = hrc::now();
            pk_treeSupportAccum += ms_d(layout_t1 - layout_t0).count()
                                 + ms_d(leafscr_t1 - leafscr_t0).count();

            // Dispatch histogram (using the eval'd layout from above)
            // Sprint 22 D0: useT2=true routes to DispatchHistogramT2 (scratch).
            auto hist_t0 = hrc::now();
            mx::array histogram =
#ifdef CATBOOST_MLX_HISTOGRAM_T2
                useT2
                ? CallDispatchHistogramT2(
                    compressedData, statArr,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    features, numUi32PerDoc, numActiveParts,
                    totalBinFeatures, numStats, numDocs, maxBlocksPerPart)
                :
#endif
                DispatchHistogram(
                    compressedData, statArr,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    features, numUi32PerDoc, numActiveParts,
                    totalBinFeatures, numStats, numDocs, maxBlocksPerPart
                );
            mx::eval(histogram);
            auto hist_t1 = hrc::now();
            pk_histAccum += ms_d(hist_t1 - hist_t0).count();

            // Find best split with sub-timing
            double suffixMs = 0.0, scoreMs = 0.0;
            auto best = FindBestSplitGPU(
                histogram, gradSumsGPU, hessSumsGPU,
                features, numActiveParts, totalBinFeatures, numStats,
                l2RegLambda, 1 /*approxDim*/, true, &suffixMs, &scoreMs
            );
            pk_suffixAccum += suffixMs;
            pk_scoreAccum  += scoreMs;

            if (!best.Defined()) break;

            // Apply split to partitions
            const TCFeature& splitFeat = features[best.FeatureId];
            ApplySplitToPartitions(
                partitions, compressedData, splitFeat,
                best.BinId, depth, numDocs, numUi32PerDoc
            );
        } else {
            // Normal (non-profiling) path — identical to original code
            auto layout = ComputePartitionLayout(partitions, numDocs, numActiveParts);

            // Dispatch histogram
            // Sprint 22 D0: useT2=true routes to DispatchHistogramT2 (scratch).
            mx::array histogram =
#ifdef CATBOOST_MLX_HISTOGRAM_T2
                useT2
                ? CallDispatchHistogramT2(
                    compressedData, statArr,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    features, numUi32PerDoc, numActiveParts,
                    totalBinFeatures, numStats, numDocs, maxBlocksPerPart)
                :
#endif
                DispatchHistogram(
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
    }

    // Per-kernel: push depth-loop accumulators
    if (pkOut) {
        pkOut->TreeSupportMs.push_back(pk_treeSupportAccum);
        pkOut->HistogramMs.push_back(pk_histAccum);
        pkOut->SuffixSumMs.push_back(pk_suffixAccum);
        pkOut->SplitScoreMs.push_back(pk_scoreAccum);
    }

    auto t2 = std::chrono::steady_clock::now();  // end of tree search stage

    // --- Estimate leaf values and apply tree ---
    auto leaf_t0 = pkOut ? hrc::now() : hrc::time_point{};

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

    // Per-kernel: record leaf estimation bucket and iter total
    if (pkOut) {
        pkOut->LeafMs.push_back(ms_d(wallEnd - leaf_t0).count());
        pkOut->IterTotalMs.push_back(ms_d(wallEnd - wallStart).count());
    }

    if (stageOut) {
        stageOut->DerivativesMs  = ms_d(t1 - t0).count();
        stageOut->TreeSearchMs   = ms_d(t2 - t1).count();
        stageOut->LeafEstimMs    = ms_d(wallEnd - t2).count();
        stageOut->IterTotalMs    = ms_d(wallEnd - wallStart).count();
    }
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

#ifdef CATBOOST_MLX_STAGE_PROFILE
    const bool doStageProfile = cfg.StageProfile;
#else
    const bool doStageProfile = false;
    if (cfg.StageProfile) {
        printf("[bench_boosting] WARNING: --stage-profile requires recompiling with -DCATBOOST_MLX_STAGE_PROFILE.\n");
        printf("  JSON output disabled for this build.\n");
    }
#endif
    std::vector<TBenchStageRecord> stageRecords;
    if (doStageProfile) stageRecords.reserve(cfg.NumIters);

    // Per-kernel timing collection (allocated always when flag set; per-iter push inside RunIteration)
    TPerKernelTimings pkTimings;
    const bool doPerKernelProfile = cfg.PerKernelProfile;
    if (doPerKernelProfile) {
        pkTimings.DerivativesMs.reserve(cfg.NumIters);
        pkTimings.TreeSupportMs.reserve(cfg.NumIters);
        pkTimings.HistogramMs.reserve(cfg.NumIters);
        pkTimings.SuffixSumMs.reserve(cfg.NumIters);
        pkTimings.SplitScoreMs.reserve(cfg.NumIters);
        pkTimings.LeafMs.reserve(cfg.NumIters);
        pkTimings.IterTotalMs.reserve(cfg.NumIters);
        printf("[per-kernel-profile] Enabled — inserting mx::eval() sync points.\n");
        printf("  WARNING: reported ms are UPPER BOUNDS (kernel overlap suppressed).\n\n");
    }

    // =========================================================================
    // Sprint 22 D0: dual-run mode
    //
    // When cfg.UseT2=true AND compiled with -DCATBOOST_MLX_HISTOGRAM_T2=1:
    //   Run T1 first (useT2=false) then T2 (useT2=true) in the same process.
    //   This cancels Metal scheduler drift (no binary re-invocation).
    //   T1 run resets cursor/partitions; T2 run resets again from the same state.
    //
    // When cfg.UseT2=false (or macro not defined): single T1 run, original behavior.
    // =========================================================================

#ifndef CATBOOST_MLX_HISTOGRAM_T2
    if (cfg.UseT2) {
        fprintf(stderr,
            "\n[FATAL] --t2 flag set but binary was NOT compiled with -DCATBOOST_MLX_HISTOGRAM_T2=1.\n"
            "  Rebuild with: clang++ ... -DCATBOOST_MLX_HISTOGRAM_T2=1 ...\n");
        return 1;
    }
#endif

    // Helper lambda to run one full pass of the boosting loop.
    // Resets cursor and partitions before the run.
    auto runBench = [&](bool useT2Variant, TPerKernelTimings& timingsOut) -> float {
        // Reset model state
        if (approxDim == 1) {
            cursor = mx::zeros({static_cast<int>(cfg.NumRows)}, mx::float32);
        } else {
            cursor = mx::zeros({static_cast<int>(approxDim), static_cast<int>(cfg.NumRows)}, mx::float32);
        }
        mx::eval(cursor);
        partitions = mx::zeros({static_cast<int>(cfg.NumRows)}, mx::uint32);
        mx::eval(partitions);

        timingsOut = TPerKernelTimings{};
        if (doPerKernelProfile) {
            timingsOut.DerivativesMs.reserve(cfg.NumIters);
            timingsOut.TreeSupportMs.reserve(cfg.NumIters);
            timingsOut.HistogramMs.reserve(cfg.NumIters);
            timingsOut.SuffixSumMs.reserve(cfg.NumIters);
            timingsOut.SplitScoreMs.reserve(cfg.NumIters);
            timingsOut.LeafMs.reserve(cfg.NumIters);
            timingsOut.IterTotalMs.reserve(cfg.NumIters);
        }

        std::vector<long long> timesUs;
        timesUs.reserve(cfg.NumIters);

        for (ui32 iter = 0; iter < cfg.NumIters; ++iter) {
            TBenchStageRecord stageRec;
            stageRec.Iter = static_cast<int>(iter);

            long long us = RunIteration(
                cursor, partitions,
                compressedData, targets,
                ds.Features,
                cfg.NumRows, ds.NumUi32PerDoc,
                1u,                    // single partition at start of each iter
                ds.TotalBinFeatures, numStats, approxDim, cfg.NumClasses,
                cfg.MaxDepth, cfg.L2RegLambda, cfg.LearningRate,
                maxBlocksPerPart,
                doStageProfile ? &stageRec : nullptr,
                doPerKernelProfile ? &timingsOut : nullptr,
                useT2Variant
            );
            timesUs.push_back(us);
            if (doStageProfile) stageRecords.push_back(stageRec);

            if (iter == 0 || (iter + 1) % 10 == 0 || iter == cfg.NumIters - 1) {
                float loss = ComputeLoss(cursor, targets, cfg.NumClasses);
                printf("  [%s] iter %4u  time=%7.1f ms  loss=%.6f\n",
                       useT2Variant ? "T2" : "T1", iter, us / 1000.0f, loss);
            }
        }

        float finalLoss = ComputeLoss(cursor, targets, cfg.NumClasses);

        // Print summary for this variant
        const char* variantLabel = useT2Variant ? "T2 sort-by-bin" : "T1 (production)";
        printf("\n================================================================\n");
        printf("Timing Summary — %s\n", variantLabel);
        printf("----------------------------------------------------------------\n");

        long long iter0Us = timesUs[0];
        double warmSumUs  = 0.0;
        long long warmMin = std::numeric_limits<long long>::max();
        long long warmMax = 0;
        for (ui32 i = 1; i < timesUs.size(); ++i) {
            warmSumUs += timesUs[i];
            warmMin = std::min(warmMin, timesUs[i]);
            warmMax = std::max(warmMax, timesUs[i]);
        }
        ui32 warmCount = (ui32)timesUs.size() - 1;
        double warmMeanUs = warmCount > 0 ? warmSumUs / warmCount : 0.0;

        printf("  iter-0 (cold start):  %8.1f ms\n", iter0Us / 1000.0f);
        if (warmCount > 0) {
            printf("  warm mean (%4u iters): %8.1f ms\n", warmCount, warmMeanUs / 1000.0f);
            printf("  warm min:             %8.1f ms\n", warmMin / 1000.0f);
            printf("  warm max:             %8.1f ms\n", warmMax / 1000.0f);
        }
        printf("\n  Final loss: %.8f\n", finalLoss);
        if (useT2Variant) {
            printf("  BENCH_FINAL_LOSS_T2=%.8f\n", finalLoss);
        } else {
            printf("  BENCH_FINAL_LOSS=%.8f\n", finalLoss);
        }
        printf("================================================================\n");

        // Store timings for per-kernel report (used after both runs finish)
        iterTimesUs = timesUs;
        return finalLoss;
    };

    // --- Run T1 (baseline, always first) ---
    printf("\n--- Running T1 (production histogram, baseline) ---\n");
    float finalLossT1 = runBench(false, pkTimings);

    // --- Run T2 (Sprint 22 D0 probe, only when --t2 is set) ---
    float finalLossT2 = finalLossT1;
    TPerKernelTimings pkTimingsT2;
    if (cfg.UseT2) {
#ifdef CATBOOST_MLX_HISTOGRAM_T2
        printf("\n--- Running T2 (sort-by-bin probe) ---\n");
        finalLossT2 = runBench(true, pkTimingsT2);
#endif
    }

    // Final per-kernel comparison (T2 D0 verdict)
    float finalLoss = finalLossT1;  // keep for compatibility with reporting below

    // Sprint 22 D0 verdict (only when T2 was also run with --per-kernel-profile)
    if (cfg.UseT2 && doPerKernelProfile &&
        !pkTimings.HistogramMs.empty() && !pkTimingsT2.HistogramMs.empty()) {

        // pkTimings = T1 run timings (set last by runBench(false,...) since T1 ran first)
        // pkTimingsT2 = T2 run timings
        // Compute 10%-trimmed means of histogram_ms for each
        auto trimMean = [](const std::vector<double>& v) -> double {
            const size_t N = v.size();
            if (N == 0) return 0.0;
            std::vector<double> warm;
            for (size_t i = 1; i < N; ++i) warm.push_back(v[i]);
            if (warm.empty()) return v[0];
            std::sort(warm.begin(), warm.end());
            size_t lo = 0, hi = warm.size();
            if (warm.size() >= 10) {
                size_t trim = warm.size() / 10;
                lo = trim; hi = warm.size() - trim;
            }
            double sum = 0.0;
            for (size_t i = lo; i < hi; ++i) sum += warm[i];
            return sum / (hi - lo);
        };

        double t1HistMs = trimMean(pkTimings.HistogramMs);
        double t2HistMs = trimMean(pkTimingsT2.HistogramMs);
        double t1IterMs = trimMean(pkTimings.IterTotalMs);
        double t2IterMs = trimMean(pkTimingsT2.IterTotalMs);
        double ratio    = (t1HistMs > 0.0) ? t2HistMs / t1HistMs : 999.0;

        // 2σ propagation: stdev of ratio = ratio × sqrt((σT2/μT2)² + (σT1/μT1)²)
        auto trimStdev = [](const std::vector<double>& v, double mean) -> double {
            const size_t N = v.size();
            std::vector<double> warm;
            for (size_t i = 1; i < N; ++i) warm.push_back(v[i]);
            if (warm.empty()) return 0.0;
            std::sort(warm.begin(), warm.end());
            size_t lo = 0, hi = warm.size();
            if (warm.size() >= 10) {
                size_t trim = warm.size() / 10;
                lo = trim; hi = warm.size() - trim;
            }
            double var = 0.0;
            for (size_t i = lo; i < hi; ++i) {
                double d = warm[i] - mean;
                var += d * d;
            }
            size_t trimN = hi - lo;
            return trimN > 1 ? std::sqrt(var / (trimN - 1)) : 0.0;
        };
        double t1HistSd = trimStdev(pkTimings.HistogramMs, t1HistMs);
        double t2HistSd = trimStdev(pkTimingsT2.HistogramMs, t2HistMs);
        double cv1 = (t1HistMs > 0.0) ? t1HistSd / t1HistMs : 0.0;
        double cv2 = (t2HistMs > 0.0) ? t2HistSd / t2HistMs : 0.0;
        double ratioSigma = ratio * std::sqrt(cv1 * cv1 + cv2 * cv2);
        double ratio2sigma = 2.0 * ratioSigma;

        // Loss sanity check
        float lossDelta = std::abs(finalLossT2 - finalLossT1);
        float lossRelDelta = (finalLossT1 > 0.0f) ? lossDelta / finalLossT1 : 0.0f;
        bool lossSane = (lossRelDelta < 0.10f);

        // Verdict
        const char* verdict;
        const char* band;
        if (ratio <= 0.45) {
            verdict = "PASS — optimistic band";
            band    = "proceed to D1 parity";
        } else if (ratio <= 0.60) {
            verdict = "PASS — conservative band";
            band    = "proceed to D1 parity; R8 projection drops to 1.37–1.51x midpoint";
        } else {
            verdict = "FAIL — T2 FALSIFIED at production shape";
            band    = "Sprint 22 pivots to tree-search restructure";
        }

        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║         Sprint 22 D0 — T2 Production Shape Verdict          ║\n");
        printf("╚══════════════════════════════════════════════════════════════╝\n");
        printf("  T1 histogram_ms   : %.3f ms  (stdev %.3f ms)\n", t1HistMs, t1HistSd);
        printf("  T2 histogram_ms   : %.3f ms  (stdev %.3f ms)\n", t2HistMs, t2HistSd);
        printf("  T2/T1 ratio       : %.4f x  (±%.4f x, 2σ)\n", ratio, ratio2sigma);
        printf("  Kill-switch bands : ≤0.45 PASS-opt | 0.45–0.60 PASS-cons | >0.60 FAIL\n");
        printf("  VERDICT           : %s\n", verdict);
        printf("  Action            : %s\n", band);
        printf("\n");
        printf("  T1 iter_total_ms  : %.3f ms\n", t1IterMs);
        printf("  T2 iter_total_ms  : %.3f ms\n", t2IterMs);
        printf("  T1 BENCH_FINAL_LOSS = %.8f\n", finalLossT1);
        printf("  T2 BENCH_FINAL_LOSS = %.8f\n", finalLossT2);
        printf("  Loss sanity       : |ΔL/L1| = %.4f%%  %s\n",
               lossRelDelta * 100.0f, lossSane ? "OK (<10%)" : "WARN (>10% — T2 correctness suspect)");
        if (!lossSane) {
            printf("  [WARN] T2 final loss differs >10%% from T1. Timing measurement may be unreliable.\n");
            printf("         Investigate T2 histogram correctness before trusting the ratio.\n");
        }
        printf("\n  S22-D0-RATIO=%.6f\n", ratio);
        printf("  S22-D0-VERDICT=%s\n", ratio <= 0.60 ? "PASS" : "FAIL");
        printf("================================================================\n");
    }

    // Per-kernel profile report
    if (doPerKernelProfile && !pkTimings.IterTotalMs.empty()) {
        // Report warm iters only: skip index 0 (cold start), same as warm_mean convention.
        // If only 1 iter was run there are no warm samples — skip.
        const size_t N = pkTimings.IterTotalMs.size();
        if (N < 2) {
            printf("\n[per-kernel-profile] Only 1 iter — no warm samples to report.\n");
        } else {
            const size_t warmN = N - 1;  // number of warm iters

            // Helper lambda: compute trimmed mean and stdev over warm iters (indices 1..N-1).
            // Uses 10% trim on each side (drop top and bottom 10% of samples) to suppress
            // Metal scheduling outliers. Trimming is applied only when warmN >= 10; below
            // that threshold a full mean is used. This is standard GPU benchmarking practice
            // for sub-ms kernels measured with wall-clock.
            auto stats = [&](const std::vector<double>& v) -> std::pair<double,double> {
                std::vector<double> warm;
                warm.reserve(warmN);
                for (size_t i = 1; i < N; ++i) warm.push_back(v[i]);
                std::sort(warm.begin(), warm.end());
                // Determine trim window
                size_t lo = 0, hi = warm.size();
                if (warm.size() >= 10) {
                    size_t trim = warm.size() / 10;  // 10% each side
                    lo = trim;
                    hi = warm.size() - trim;
                }
                const size_t trimN = hi - lo;
                double sum = 0.0;
                for (size_t i = lo; i < hi; ++i) sum += warm[i];
                double mean = sum / trimN;
                double var  = 0.0;
                for (size_t i = lo; i < hi; ++i) {
                    double d = warm[i] - mean;
                    var += d * d;
                }
                double stdev = trimN > 1 ? std::sqrt(var / (trimN - 1)) : 0.0;
                return {mean, stdev};
            };

            auto [d_mean,  d_sd]  = stats(pkTimings.DerivativesMs);
            auto [ts_mean, ts_sd] = stats(pkTimings.TreeSupportMs);
            auto [h_mean,  h_sd]  = stats(pkTimings.HistogramMs);
            auto [sf_mean, sf_sd] = stats(pkTimings.SuffixSumMs);
            auto [sc_mean, sc_sd] = stats(pkTimings.SplitScoreMs);
            auto [lf_mean, lf_sd] = stats(pkTimings.LeafMs);
            auto [it_mean, it_sd] = stats(pkTimings.IterTotalMs);

            double sumKernels = d_mean + ts_mean + h_mean + sf_mean + sc_mean + lf_mean;
            double deltaPct   = it_mean > 0.0 ? 100.0 * (sumKernels - it_mean) / it_mean : 0.0;

            auto pct = [](double mean, double sd) -> double {
                return mean > 0.0 ? 100.0 * sd / mean : 0.0;
            };

            printf("\n=== Per-kernel profile (--per-kernel-profile; UPPER BOUNDS due to sync) ===\n");
            printf("WARNING: mx::eval() sync points disable MLX kernel overlap. Reported\n");
            printf("         per-kernel ms are UPPER BOUNDS on non-sync production timing.\n");
            printf("         Stats: 10%%-trimmed mean/stdev (Metal scheduling jitter suppressed)\n");
            printf("         Warm iters = %zu (iter-0 excluded)\n\n", warmN);

            // Print each bucket.
            // For sub-millisecond buckets (mean < 2 ms), stdev% > 5% is expected:
            // Apple Metal command-buffer submit latency jitter is ~20-100 µs,
            // which dominates a 0.5-1.0 ms measurement regardless of sample count.
            // We annotate these as [wall-clock floor] rather than a data quality warning.
            auto printBucket = [&](const char* label, double mean, double sd) {
                double sdpct = pct(mean, sd);
                printf("  %-16s mean=%7.3f ms   stdev=%6.3f ms  (%4.1f%%)",
                       label, mean, sd, sdpct);
                if (sdpct > 5.0) {
                    if (mean < 2.0) {
                        printf("  [wall-clock floor: sub-ms jitter]");
                    } else {
                        printf("  [WARN: stdev > 5%%]");
                    }
                }
                printf("\n");
            };

            printBucket("derivatives",    d_mean,  d_sd);
            printBucket("tree_support",   ts_mean, ts_sd);
            printBucket("histogram",      h_mean,  h_sd);
            printBucket("suffix_sum",     sf_mean, sf_sd);
            printBucket("split_score",    sc_mean, sc_sd);
            printBucket("leaf_estimation",lf_mean, lf_sd);
            printf("  ------ sum-of-per-kernel=%7.3f ms  vs iter_total=%7.3f ms"
                   "  (delta=%+.3f ms, %+.1f%%)\n",
                   sumKernels, it_mean, sumKernels - it_mean, deltaPct);

            if (std::abs(deltaPct) > 5.0) {
                printf("  [WARN: |delta| > 5%% — possible missing dispatch boundary]\n");
            }
        }
    }

    // Stage profile JSON dump
    if (doStageProfile && !stageRecords.empty()) {
#ifdef CATBOOST_MLX_STAGE_PROFILE
        const char* baseDir = ".cache/profiling/sprint16";
        try { std::filesystem::create_directories(baseDir); } catch (...) {}
        auto ts = static_cast<long long>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
        std::string outPath = std::string(baseDir) + "/stage_times_bench_" +
                              std::to_string(ts) + ".json";
        std::ostringstream metaOs;
        metaOs << "{"
               << "\"source\":\"bench_boosting\","
               << "\"build\":\"CATBOOST_MLX_STAGE_PROFILE=ON\","
               << "\"rows\":" << cfg.NumRows
               << ",\"features\":" << cfg.NumFeatures
               << ",\"classes\":" << cfg.NumClasses
               << ",\"depth\":" << cfg.MaxDepth
               << ",\"iters\":" << cfg.NumIters
               << ",\"bins\":" << cfg.NumBins
               << "}";
        if (WriteBenchStageJson(stageRecords, outPath, metaOs.str())) {
            printf("Stage profile written to: %s\n", outPath.c_str());
        }
#else
        printf("[stage-profile] Rebuild with -DCATBOOST_MLX_STAGE_PROFILE for JSON output.\n");
        printf("  --stage-profile flag is present but CATBOOST_MLX_STAGE_PROFILE is not defined.\n");
#endif
    }

    return 0;
}
