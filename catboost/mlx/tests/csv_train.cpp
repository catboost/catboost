// CatBoost-MLX standalone CSV training tool.
// Reads a CSV file, quantizes features, and trains a GBDT model on Apple Silicon GPU.
//
// Usage: ./csv_train <file.csv> [options]
//   --iterations N     Number of boosting iterations (default: 100)
//   --depth D          Max tree depth (default: 6)
//   --lr RATE          Learning rate (default: 0.1)
//   --l2 LAMBDA        L2 regularization (default: 3.0)
//   --loss TYPE        Loss function: rmse, logloss, multiclass, auto (default: auto)
//   --bins B           Max quantization bins per feature (default: 255)
//   --target-col N     0-based column index for target (default: last column)
//   --verbose          Print per-iteration loss
//
// Compile:
//   clang++ -std=c++17 -O2 -I. \
//     -I/opt/homebrew/Cellar/mlx/0.31.1/include \
//     -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
//     -framework Metal -framework Foundation -Wno-c++20-extensions \
//     catboost/mlx/tests/csv_train.cpp -o csv_train
//
// Examples:
//   ./csv_train iris.csv --loss multiclass --iterations 200
//   ./csv_train housing.csv --loss rmse --depth 4 --lr 0.05
//   ./csv_train breast_cancer.csv --loss logloss

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <catboost/mlx/kernels/kernel_sources.h>

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>

namespace mx = mlx::core;
using ui32 = uint32_t;

// ============================================================================
// Structures (mirrored from gpu_structures.h for standalone compilation)
// ============================================================================

struct TCFeature {
    uint64_t Offset;
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
    float Score = 1e30f;
    float Gain = 1e30f;
    bool Defined() const { return FeatureId != static_cast<ui32>(-1); }
};

struct TObliviousSplitLevel {
    ui32 FeatureColumnIdx;
    ui32 Shift;
    ui32 Mask;
    ui32 BinThreshold;
};

struct TPartitionStatistics {
    double Sum = 0.0;
    double Weight = 0.0;
    double Count = 0.0;
};

// ============================================================================
// CLI argument parsing
// ============================================================================

struct TConfig {
    std::string CsvPath;
    ui32 NumIterations = 100;
    ui32 MaxDepth = 6;
    float LearningRate = 0.1f;
    float L2RegLambda = 3.0f;
    ui32 MaxBins = 255;
    int TargetCol = -1;  // -1 = last column
    std::string LossType = "auto";
    bool Verbose = false;
};

TConfig ParseArgs(int argc, char** argv) {
    TConfig config;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file.csv> [--iterations N] [--depth D] [--lr RATE] "
                "[--l2 LAMBDA] [--loss TYPE] [--bins B] [--target-col N] [--verbose]\n", argv[0]);
        exit(1);
    }
    config.CsvPath = argv[1];
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) config.NumIterations = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) config.MaxDepth = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) config.LearningRate = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--l2") == 0 && i + 1 < argc) config.L2RegLambda = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--loss") == 0 && i + 1 < argc) config.LossType = argv[++i];
        else if (strcmp(argv[i], "--bins") == 0 && i + 1 < argc) config.MaxBins = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--target-col") == 0 && i + 1 < argc) config.TargetCol = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0) config.Verbose = true;
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); exit(1); }
    }
    return config;
}

// ============================================================================
// CSV loading
// ============================================================================

struct TDataset {
    std::vector<std::vector<float>> Features;  // [numFeatures][numDocs]
    std::vector<float> Targets;                 // [numDocs]
    ui32 NumDocs = 0;
    ui32 NumFeatures = 0;
    std::vector<std::string> FeatureNames;
};

bool IsNumber(const std::string& s) {
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

TDataset LoadCSV(const std::string& path, int targetCol) {
    TDataset ds;
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file: %s\n", path.c_str());
        exit(1);
    }

    std::string line;
    std::vector<std::vector<float>> rows;
    bool hasHeader = false;
    int numCols = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        // Trim trailing carriage return
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) {
            // Trim whitespace
            size_t start = cell.find_first_not_of(" \t");
            size_t end = cell.find_last_not_of(" \t");
            if (start != std::string::npos) cell = cell.substr(start, end - start + 1);
            cells.push_back(cell);
        }

        if (numCols == 0) {
            numCols = cells.size();
            // Check if first row is a header
            if (!cells.empty() && !IsNumber(cells[0])) {
                hasHeader = true;
                ds.FeatureNames = cells;
                continue;
            }
        }

        if (static_cast<int>(cells.size()) != numCols) continue;  // skip malformed rows

        std::vector<float> row(numCols);
        bool valid = true;
        for (int i = 0; i < numCols; ++i) {
            try {
                row[i] = std::stof(cells[i]);
            } catch (...) {
                valid = false;
                break;
            }
        }
        if (valid) rows.push_back(row);
    }

    if (rows.empty()) {
        fprintf(stderr, "Error: No valid data rows in %s\n", path.c_str());
        exit(1);
    }

    ds.NumDocs = rows.size();

    // Determine target column
    int tgtCol = (targetCol >= 0) ? targetCol : (numCols - 1);
    if (tgtCol < 0 || tgtCol >= numCols) {
        fprintf(stderr, "Error: Invalid target column %d (data has %d columns)\n", tgtCol, numCols);
        exit(1);
    }

    // Split into features and targets
    ds.NumFeatures = numCols - 1;
    ds.Features.resize(ds.NumFeatures);
    for (ui32 f = 0; f < ds.NumFeatures; ++f) {
        ds.Features[f].resize(ds.NumDocs);
    }
    ds.Targets.resize(ds.NumDocs);

    for (ui32 d = 0; d < ds.NumDocs; ++d) {
        ds.Targets[d] = rows[d][tgtCol];
        ui32 fIdx = 0;
        for (int c = 0; c < numCols; ++c) {
            if (c == tgtCol) continue;
            ds.Features[fIdx++][d] = rows[d][c];
        }
    }

    return ds;
}

// ============================================================================
// Quantization
// ============================================================================

struct TQuantization {
    std::vector<std::vector<float>> Borders;    // [numFeatures] → sorted border values
    std::vector<std::vector<uint8_t>> BinnedFeatures;  // [numFeatures][numDocs] bin indices
};

TQuantization QuantizeFeatures(const TDataset& ds, ui32 maxBins) {
    TQuantization q;
    q.Borders.resize(ds.NumFeatures);
    q.BinnedFeatures.resize(ds.NumFeatures);

    for (ui32 f = 0; f < ds.NumFeatures; ++f) {
        // Sort feature values to compute equal-frequency borders
        std::vector<float> sorted = ds.Features[f];
        std::sort(sorted.begin(), sorted.end());

        // Remove duplicates for border computation
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

        ui32 numBorders = std::min(static_cast<ui32>(sorted.size()) - 1, maxBins - 1);
        q.Borders[f].resize(numBorders);

        for (ui32 b = 0; b < numBorders; ++b) {
            // Equal-frequency: pick percentile borders
            float frac = static_cast<float>(b + 1) / static_cast<float>(numBorders + 1);
            ui32 idx = static_cast<ui32>(frac * (sorted.size() - 1));
            idx = std::min(idx, static_cast<ui32>(sorted.size()) - 1);
            // Border = midpoint between sorted[idx] and sorted[idx+1] if possible
            if (idx + 1 < sorted.size()) {
                q.Borders[f][b] = 0.5f * (sorted[idx] + sorted[idx + 1]);
            } else {
                q.Borders[f][b] = sorted[idx];
            }
        }

        // Ensure borders are strictly increasing
        for (ui32 b = 1; b < numBorders; ++b) {
            if (q.Borders[f][b] <= q.Borders[f][b - 1]) {
                q.Borders[f][b] = q.Borders[f][b - 1] + 1e-6f;
            }
        }

        // Quantize: map each value to a bin using upper_bound
        q.BinnedFeatures[f].resize(ds.NumDocs);
        for (ui32 d = 0; d < ds.NumDocs; ++d) {
            auto it = std::upper_bound(q.Borders[f].begin(), q.Borders[f].end(), ds.Features[f][d]);
            q.BinnedFeatures[f][d] = static_cast<uint8_t>(it - q.Borders[f].begin());
        }
    }

    return q;
}

// ============================================================================
// Feature packing into compressed uint32 format
// ============================================================================

struct TPackedData {
    std::vector<uint32_t> Data;  // [numDocs * numUi32PerDoc]
    std::vector<TCFeature> Features;
    ui32 NumUi32PerDoc;
    ui32 TotalBinFeatures;
};

TPackedData PackFeatures(const TQuantization& q, ui32 numDocs, ui32 numFeatures) {
    TPackedData packed;
    packed.NumUi32PerDoc = (numFeatures + 3) / 4;
    packed.Data.resize(numDocs * packed.NumUi32PerDoc, 0);
    packed.TotalBinFeatures = 0;

    for (ui32 f = 0; f < numFeatures; ++f) {
        ui32 wordIdx = f / 4;
        ui32 posInWord = f % 4;
        ui32 shift = (3 - posInWord) * 8;
        ui32 mask = 0xFF << shift;
        ui32 folds = q.Borders[f].size();

        TCFeature feat;
        feat.Offset = wordIdx;
        feat.Mask = mask;
        feat.Shift = shift;
        feat.FirstFoldIndex = packed.TotalBinFeatures;
        feat.Folds = folds;
        feat.OneHotFeature = false;
        feat.SkipFirstBinInScoreCount = false;
        packed.Features.push_back(feat);
        packed.TotalBinFeatures += folds;

        // Pack bin values into compressed data
        for (ui32 d = 0; d < numDocs; ++d) {
            packed.Data[d * packed.NumUi32PerDoc + wordIdx] |=
                (static_cast<uint32_t>(q.BinnedFeatures[f][d]) << shift);
        }
    }

    return packed;
}

// ============================================================================
// Loss function auto-detection
// ============================================================================

std::string DetectLossType(const std::vector<float>& targets) {
    // Check if targets are all integers
    bool allInteger = true;
    float minTarget = targets[0], maxTarget = targets[0];
    std::vector<float> uniqueTargets;

    for (float t : targets) {
        if (t != std::floor(t)) allInteger = false;
        minTarget = std::min(minTarget, t);
        maxTarget = std::max(maxTarget, t);
    }

    if (!allInteger) return "rmse";

    // Count unique integer values
    std::vector<float> sorted = targets;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    if (sorted.size() == 2 && minTarget == 0.0f && maxTarget == 1.0f) {
        return "logloss";
    }

    // Check if targets are 0..K-1
    if (minTarget == 0.0f && maxTarget == static_cast<float>(sorted.size() - 1)) {
        bool consecutive = true;
        for (ui32 i = 0; i < sorted.size(); ++i) {
            if (sorted[i] != static_cast<float>(i)) { consecutive = false; break; }
        }
        if (consecutive && sorted.size() > 2) return "multiclass";
        if (consecutive && sorted.size() == 2) return "logloss";
    }

    return "rmse";
}

// ============================================================================
// Partition layout computation
// ============================================================================

struct TPartitionLayout {
    mx::array DocIndices;
    mx::array PartOffsets;
    mx::array PartSizes;
};

TPartitionLayout ComputePartitionLayout(const mx::array& partitions, ui32 numDocs, ui32 numPartitions) {
    mx::eval(partitions);
    const uint32_t* partsPtr = partitions.data<uint32_t>();

    std::vector<ui32> partSizes(numPartitions, 0);
    for (ui32 d = 0; d < numDocs; ++d) {
        ui32 p = partsPtr[d];
        if (p < numPartitions) partSizes[p]++;
    }

    std::vector<ui32> partOffsets(numPartitions, 0);
    for (ui32 i = 1; i < numPartitions; ++i)
        partOffsets[i] = partOffsets[i-1] + partSizes[i-1];

    std::vector<ui32> docIndices(numDocs);
    std::vector<ui32> writePos = partOffsets;
    for (ui32 d = 0; d < numDocs; ++d) {
        ui32 p = partsPtr[d];
        if (p < numPartitions) docIndices[writePos[p]++] = d;
    }

    return {
        mx::array(reinterpret_cast<const int32_t*>(docIndices.data()), {static_cast<int>(numDocs)}, mx::uint32),
        mx::array(reinterpret_cast<const int32_t*>(partOffsets.data()), {static_cast<int>(numPartitions)}, mx::uint32),
        mx::array(reinterpret_cast<const int32_t*>(partSizes.data()), {static_cast<int>(numPartitions)}, mx::uint32)
    };
}

// ============================================================================
// Histogram dispatch
// ============================================================================

mx::array DispatchHistogram(
    const mx::array& compressedData,
    const mx::array& stats,
    const mx::array& docIndices,
    const mx::array& partOffsets,
    const mx::array& partSizes,
    const std::vector<TCFeature>& features,
    ui32 lineSize,
    ui32 totalBinFeatures,
    ui32 numPartitions,
    ui32 numDocs
) {
    const ui32 numStats = 2;
    const ui32 numFeatures = features.size();
    const ui32 numFeatureGroups = (numFeatures + 3) / 4;

    mx::Shape histShape = {static_cast<int>(numPartitions * numStats * totalBinFeatures)};
    mx::array histogram = mx::zeros(histShape, mx::float32);

    for (ui32 groupIdx = 0; groupIdx < numFeatureGroups; ++groupIdx) {
        const ui32 featureStart = groupIdx * 4;
        const ui32 featuresInGroup = std::min(4u, numFeatures - featureStart);

        std::vector<ui32> foldCountsVec(4, 0);
        std::vector<ui32> firstFoldVec(4, 0);
        for (ui32 f = 0; f < featuresInGroup; ++f) {
            foldCountsVec[f] = features[featureStart + f].Folds;
            firstFoldVec[f] = features[featureStart + f].FirstFoldIndex;
        }

        auto foldCountsArr = mx::array(reinterpret_cast<const int32_t*>(foldCountsVec.data()), {4}, mx::uint32);
        auto firstFoldArr = mx::array(reinterpret_cast<const int32_t*>(firstFoldVec.data()), {4}, mx::uint32);

        using namespace NCatboostMlx;
        auto kernel = mx::fast::metal_kernel(
            "histogram_one_byte_features",
            {"compressedIndex", "stats", "docIndices",
             "partOffsets", "partSizes",
             "featureColumnIdx", "lineSize", "maxBlocksPerPart",
             "foldCounts", "firstFoldIndices",
             "totalBinFeatures", "numStats", "totalNumDocs"},
            {"histogram"},
            KernelSources::kHistOneByteSource,
            KernelSources::kHistHeader,
            true, false
        );

        auto result = kernel(
            {mx::reshape(compressedData, {-1}), stats, docIndices,
             partOffsets, partSizes,
             mx::array(static_cast<uint32_t>(groupIdx), mx::uint32),
             mx::array(static_cast<uint32_t>(lineSize), mx::uint32),
             mx::array(static_cast<uint32_t>(1), mx::uint32),
             foldCountsArr, firstFoldArr,
             mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32),
             mx::array(static_cast<uint32_t>(numStats), mx::uint32),
             mx::array(static_cast<uint32_t>(numDocs), mx::uint32)},
            {histShape}, {mx::float32},
            std::make_tuple(256, static_cast<int>(numPartitions), 2),
            std::make_tuple(256, 1, 1),
            {}, 0.0f, false, mx::Device::gpu
        );

        if (groupIdx == 0) histogram = result[0];
        else histogram = mx::add(histogram, result[0]);
    }

    mx::eval(histogram);
    return histogram;
}

// ============================================================================
// Split finding (multi-dim aware)
// ============================================================================

TBestSplitProperties FindBestSplit(
    const std::vector<std::vector<float>>& perDimHist,  // [K][numPartitions * 2 * totalBinFeatures]
    const std::vector<std::vector<TPartitionStatistics>>& perDimPartStats,  // [K][numPartitions]
    const std::vector<TCFeature>& features,
    ui32 totalBinFeatures,
    float l2RegLambda,
    ui32 numPartitions
) {
    TBestSplitProperties bestSplit;
    const ui32 K = perDimHist.size();
    float bestGain = -std::numeric_limits<float>::infinity();

    // The Metal histogram kernel produces per-bin sums with a +1 offset:
    //   hist[firstFold + b] = sum of docs where featureValue == b+1
    //
    // For a split at threshold b (featureValue > b → right):
    //   Left side = featureValue ∈ {0, 1, ..., b}
    //   Right side = featureValue ∈ {b+1, b+2, ..., folds}
    //
    // We compute sumRight as the suffix sum of hist[b..folds-1],
    // and sumLeft = totalSum - sumRight.

    for (ui32 featIdx = 0; featIdx < features.size(); ++featIdx) {
        const auto& feat = features[featIdx];

        for (ui32 bin = 0; bin < feat.Folds; ++bin) {
            float totalGain = 0.0f;

            for (ui32 p = 0; p < numPartitions; ++p) {
                for (ui32 k = 0; k < K; ++k) {
                    const float* histData = perDimHist[k].data() + p * 2 * totalBinFeatures;

                    // Suffix sum: sum of hist[bin..folds-1] = sum(featureValue ∈ {bin+1,...,folds})
                    // This is the RIGHT side for threshold = bin
                    float sumRight = 0.0f;
                    float weightRight = 0.0f;
                    for (ui32 b = bin; b < feat.Folds; ++b) {
                        sumRight += histData[feat.FirstFoldIndex + b];
                        weightRight += histData[totalBinFeatures + feat.FirstFoldIndex + b];
                    }

                    float totalSum = static_cast<float>(perDimPartStats[k][p].Sum);
                    float totalWeight = static_cast<float>(perDimPartStats[k][p].Weight);
                    float sumLeft = totalSum - sumRight;
                    float weightLeft = totalWeight - weightRight;

                    // Skip partitions where one side is empty — zero gain contribution
                    if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;

                    totalGain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda)
                               + (sumRight * sumRight) / (weightRight + l2RegLambda)
                               - (totalSum * totalSum) / (totalWeight + l2RegLambda);
                }
            }

            if (totalGain > bestGain) {
                bestGain = totalGain;
                bestSplit.FeatureId = featIdx;
                bestSplit.BinId = bin;
                bestSplit.Gain = totalGain;
                bestSplit.Score = -totalGain;
            }
        }
    }
    return bestSplit;
}

// ============================================================================
// Softmax helpers
// ============================================================================

struct TSoftmaxResult {
    mx::array Probs;  // [K, numDocs]
};

TSoftmaxResult ComputeSoftmax(const mx::array& cursor, int K, int numDocs) {
    auto maxC = mx::maximum(mx::max(cursor, 0), mx::array(0.0f));  // [numDocs]
    auto expC = mx::exp(mx::subtract(cursor, mx::reshape(maxC, {1, numDocs})));  // [K, numDocs]
    auto expImp = mx::exp(mx::negative(maxC));  // [numDocs]
    auto sumExp = mx::add(mx::sum(expC, 0), expImp);  // [numDocs]
    auto probs = mx::divide(expC, mx::reshape(sumExp, {1, numDocs}));  // [K, numDocs]
    return {probs};
}

// ============================================================================
// Main training loop
// ============================================================================

int main(int argc, char** argv) {
    auto config = ParseArgs(argc, argv);

    printf("CatBoost-MLX CSV Training Tool\n");
    printf("==============================\n");

    // Load data
    auto ds = LoadCSV(config.CsvPath, config.TargetCol);
    printf("Loaded: %u rows, %u features from %s\n", ds.NumDocs, ds.NumFeatures, config.CsvPath.c_str());

    // Detect or validate loss type
    std::string lossType = config.LossType;
    if (lossType == "auto") {
        lossType = DetectLossType(ds.Targets);
        printf("Auto-detected loss: %s\n", lossType.c_str());
    }

    // Determine approx dimension
    ui32 approxDim = 1;
    ui32 numClasses = 0;
    if (lossType == "multiclass") {
        float maxTarget = *std::max_element(ds.Targets.begin(), ds.Targets.end());
        numClasses = static_cast<ui32>(maxTarget) + 1;
        approxDim = numClasses - 1;
        printf("MultiClass: %u classes, approxDim=%u\n", numClasses, approxDim);
    }

    // Quantize
    auto quant = QuantizeFeatures(ds, config.MaxBins);
    ui32 totalBorders = 0;
    for (ui32 f = 0; f < ds.NumFeatures; ++f) totalBorders += quant.Borders[f].size();
    printf("Quantized: %u total borders across %u features\n", totalBorders, ds.NumFeatures);
    // Pack features
    auto packed = PackFeatures(quant, ds.NumDocs, ds.NumFeatures);
    printf("Packed: %u uint32 words per doc, %u total bin-features\n",
           packed.NumUi32PerDoc, packed.TotalBinFeatures);

    if (packed.TotalBinFeatures == 0) {
        fprintf(stderr, "Error: No valid features after quantization\n");
        return 1;
    }

    // Transfer to GPU
    auto compressedData = mx::array(
        reinterpret_cast<const int32_t*>(packed.Data.data()),
        {static_cast<int>(ds.NumDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32
    );
    auto targetsArr = mx::array(ds.Targets.data(), {static_cast<int>(ds.NumDocs)}, mx::float32);
    auto weights = mx::ones({static_cast<int>(ds.NumDocs)}, mx::float32);

    // Initialize cursor
    mx::array cursor = (approxDim == 1)
        ? mx::zeros({static_cast<int>(ds.NumDocs)}, mx::float32)
        : mx::zeros({static_cast<int>(approxDim), static_cast<int>(ds.NumDocs)}, mx::float32);

    printf("\nTraining: %u iterations, depth=%u, lr=%.4f, l2=%.2f, loss=%s\n",
           config.NumIterations, config.MaxDepth, config.LearningRate, config.L2RegLambda, lossType.c_str());
    printf("---\n");

    auto startTime = std::chrono::steady_clock::now();
    ui32 treesBuilt = 0;

    for (ui32 iter = 0; iter < config.NumIterations; ++iter) {
        auto iterStart = std::chrono::steady_clock::now();

        // Step 1: Compute gradients per dimension
        std::vector<mx::array> dimGrads, dimHess;
        dimGrads.reserve(approxDim);
        dimHess.reserve(approxDim);
        for (ui32 k = 0; k < approxDim; ++k) {
            dimGrads.push_back(mx::zeros({static_cast<int>(ds.NumDocs)}, mx::float32));
            dimHess.push_back(mx::zeros({static_cast<int>(ds.NumDocs)}, mx::float32));
        }

        if (lossType == "rmse") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(ds.NumDocs)});
            dimGrads[0] = mx::subtract(flatCursor, targetsArr);
            dimHess[0] = mx::ones({static_cast<int>(ds.NumDocs)}, mx::float32);
        } else if (lossType == "logloss") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(ds.NumDocs)});
            auto sigmoid = mx::sigmoid(flatCursor);
            dimGrads[0] = mx::subtract(sigmoid, targetsArr);
            dimHess[0] = mx::maximum(
                mx::multiply(sigmoid, mx::subtract(mx::array(1.0f), sigmoid)),
                mx::array(1e-16f)
            );
        } else {  // multiclass
            auto sm = ComputeSoftmax(cursor, approxDim, ds.NumDocs);
            auto targetInt = mx::astype(targetsArr, mx::uint32);
            mx::eval(sm.Probs);

            for (ui32 k = 0; k < approxDim; ++k) {
                auto isClass = mx::astype(
                    mx::equal(targetInt, mx::array(static_cast<uint32_t>(k))),
                    mx::float32
                );
                auto probK = mx::reshape(
                    mx::slice(sm.Probs, {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(ds.NumDocs)}),
                    {static_cast<int>(ds.NumDocs)}
                );
                dimGrads[k] = mx::subtract(probK, isClass);
                dimHess[k] = mx::maximum(
                    mx::multiply(probK, mx::subtract(mx::array(1.0f), probK)),
                    mx::array(1e-16f)
                );
            }
        }

        for (ui32 k = 0; k < approxDim; ++k) mx::eval({dimGrads[k], dimHess[k]});

        // Step 2: Greedy tree structure search
        auto partitions = mx::zeros({static_cast<int>(ds.NumDocs)}, mx::uint32);
        std::vector<TObliviousSplitLevel> splits;
        std::vector<TBestSplitProperties> splitProps;

        for (ui32 depth = 0; depth < config.MaxDepth; ++depth) {
            ui32 numPartitions = 1u << depth;
            auto layout = ComputePartitionLayout(partitions, ds.NumDocs, numPartitions);

            // Compute per-dim histograms and partition stats
            std::vector<std::vector<float>> perDimHistData(approxDim);
            std::vector<std::vector<TPartitionStatistics>> perDimPartStats(approxDim);

            mx::eval(partitions);
            const uint32_t* partsPtr = partitions.data<uint32_t>();

            for (ui32 k = 0; k < approxDim; ++k) {
                auto statsK = mx::concatenate({
                    mx::reshape(dimGrads[k], {1, static_cast<int>(ds.NumDocs)}),
                    mx::reshape(dimHess[k], {1, static_cast<int>(ds.NumDocs)})
                }, 0);
                statsK = mx::reshape(statsK, {static_cast<int>(2 * ds.NumDocs)});

                auto hist = DispatchHistogram(
                    compressedData, statsK,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    packed.Features, packed.NumUi32PerDoc,
                    packed.TotalBinFeatures, numPartitions, ds.NumDocs
                );
                mx::eval(hist);
                const float* hData = hist.data<float>();
                perDimHistData[k].assign(hData, hData + numPartitions * 2 * packed.TotalBinFeatures);

                mx::eval(dimGrads[k]);
                mx::eval(dimHess[k]);
                const float* gPtr = dimGrads[k].data<float>();
                const float* hPtr = dimHess[k].data<float>();

                perDimPartStats[k].resize(numPartitions);
                for (ui32 d = 0; d < ds.NumDocs; ++d) {
                    ui32 p = partsPtr[d];
                    if (p < numPartitions) {
                        perDimPartStats[k][p].Sum += gPtr[d];
                        perDimPartStats[k][p].Weight += hPtr[d];
                    }
                }
            }

            auto bestSplit = FindBestSplit(
                perDimHistData, perDimPartStats,
                packed.Features, packed.TotalBinFeatures,
                config.L2RegLambda, numPartitions
            );

            if (!bestSplit.Defined()) break;

            // Record split
            const auto& feat = packed.Features[bestSplit.FeatureId];
            TObliviousSplitLevel split;
            split.FeatureColumnIdx = static_cast<ui32>(feat.Offset);
            split.Shift = feat.Shift;
            split.Mask = feat.Mask >> feat.Shift;
            split.BinThreshold = bestSplit.BinId;
            splits.push_back(split);
            splitProps.push_back(bestSplit);

            // Update partitions
            auto column = mx::slice(compressedData,
                {0, static_cast<int>(split.FeatureColumnIdx)},
                {static_cast<int>(ds.NumDocs), static_cast<int>(split.FeatureColumnIdx + 1)});
            column = mx::reshape(column, {static_cast<int>(ds.NumDocs)});
            auto featureValues = mx::bitwise_and(
                mx::right_shift(column, mx::array(static_cast<uint32_t>(split.Shift), mx::uint32)),
                mx::array(static_cast<uint32_t>(split.Mask), mx::uint32));
            auto goRight = mx::greater(featureValues, mx::array(static_cast<uint32_t>(split.BinThreshold), mx::uint32));
            auto bits = mx::left_shift(mx::astype(goRight, mx::uint32), mx::array(static_cast<uint32_t>(depth), mx::uint32));
            partitions = mx::bitwise_or(partitions, bits);
            mx::eval(partitions);
        }

        if (splits.empty()) {
            printf("iter=%u: no valid split, stopping\n", iter);
            break;
        }

        // Step 3: Estimate leaf values
        ui32 numLeaves = 1u << splits.size();
        mx::eval(partitions);
        const uint32_t* leafAssign = partitions.data<uint32_t>();

        mx::array leafValues = mx::zeros({1}, mx::float32); // placeholder, overwritten below
        if (approxDim == 1) {
            mx::eval(dimGrads[0]); mx::eval(dimHess[0]);
            const float* gp = dimGrads[0].data<float>();
            const float* hp = dimHess[0].data<float>();

            std::vector<float> gSums(numLeaves, 0.0f), hSums(numLeaves, 0.0f);
            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                ui32 leaf = leafAssign[d];
                if (leaf < numLeaves) { gSums[leaf] += gp[d]; hSums[leaf] += hp[d]; }
            }
            std::vector<float> lv(numLeaves);
            for (ui32 i = 0; i < numLeaves; ++i)
                lv[i] = -config.LearningRate * gSums[i] / (hSums[i] + config.L2RegLambda);

            leafValues = mx::array(lv.data(), {static_cast<int>(numLeaves)}, mx::float32);
        } else {
            std::vector<float> interleaved(numLeaves * approxDim, 0.0f);
            for (ui32 k = 0; k < approxDim; ++k) {
                mx::eval(dimGrads[k]); mx::eval(dimHess[k]);
                const float* gp = dimGrads[k].data<float>();
                const float* hp = dimHess[k].data<float>();

                std::vector<float> gSums(numLeaves, 0.0f), hSums(numLeaves, 0.0f);
                for (ui32 d = 0; d < ds.NumDocs; ++d) {
                    ui32 leaf = leafAssign[d];
                    if (leaf < numLeaves) { gSums[leaf] += gp[d]; hSums[leaf] += hp[d]; }
                }
                for (ui32 i = 0; i < numLeaves; ++i)
                    interleaved[i * approxDim + k] = -config.LearningRate * gSums[i] / (hSums[i] + config.L2RegLambda);
            }
            leafValues = mx::array(interleaved.data(),
                {static_cast<int>(numLeaves), static_cast<int>(approxDim)}, mx::float32);
        }

        // Step 4: Apply tree
        auto docLeafValues = mx::take(leafValues, mx::astype(partitions, mx::int32), 0);
        if (approxDim > 1) {
            docLeafValues = mx::transpose(docLeafValues);  // [K, numDocs]
            cursor = mx::add(cursor, docLeafValues);
        } else {
            cursor = mx::add(mx::reshape(cursor, {static_cast<int>(ds.NumDocs)}), docLeafValues);
        }
        mx::eval(cursor);
        treesBuilt++;

        // Step 5: Report loss
        auto iterEnd = std::chrono::steady_clock::now();
        auto iterMs = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart).count();

        if (config.Verbose || iter % 10 == 0 || iter == config.NumIterations - 1) {
            float lossVal = 0.0f;
            if (lossType == "rmse") {
                auto diff = mx::subtract(mx::reshape(cursor, {static_cast<int>(ds.NumDocs)}), targetsArr);
                auto loss = mx::sqrt(mx::mean(mx::multiply(diff, diff)));
                mx::eval(loss);
                lossVal = loss.item<float>();
            } else if (lossType == "logloss") {
                auto sig = mx::sigmoid(mx::reshape(cursor, {static_cast<int>(ds.NumDocs)}));
                auto eps = mx::array(1e-15f);
                auto loss = mx::negative(mx::mean(mx::add(
                    mx::multiply(targetsArr, mx::log(mx::add(sig, eps))),
                    mx::multiply(mx::subtract(mx::array(1.0f), targetsArr),
                                 mx::log(mx::add(mx::subtract(mx::array(1.0f), sig), eps)))
                )));
                mx::eval(loss);
                lossVal = loss.item<float>();
            } else {
                auto sm = ComputeSoftmax(cursor, approxDim, ds.NumDocs);
                auto targetInt = mx::astype(targetsArr, mx::int32);
                auto probTarget = mx::zeros({static_cast<int>(ds.NumDocs)}, mx::float32);
                for (ui32 k = 0; k < approxDim; ++k) {
                    auto isClass = mx::astype(mx::equal(targetInt, mx::array(static_cast<int>(k))), mx::float32);
                    auto probK = mx::reshape(
                        mx::slice(sm.Probs, {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(ds.NumDocs)}),
                        {static_cast<int>(ds.NumDocs)});
                    probTarget = mx::add(probTarget, mx::multiply(isClass, probK));
                }
                auto isLast = mx::astype(mx::equal(targetInt, mx::array(static_cast<int>(approxDim))), mx::float32);
                auto probImp = mx::subtract(mx::array(1.0f), mx::sum(sm.Probs, 0));
                probTarget = mx::add(probTarget, mx::multiply(isLast, probImp));
                auto loss = mx::negative(mx::mean(mx::log(mx::add(probTarget, mx::array(1e-15f)))));
                mx::eval(loss);
                lossVal = loss.item<float>();
            }

            printf("iter=%u  trees=%u  depth=%zu  loss=%.6f  time=%lldms\n",
                   iter, treesBuilt, splits.size(), lossVal, iterMs);
        }
    }

    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count();

    printf("---\n");
    printf("Training complete: %u trees in %.2fs\n", treesBuilt, totalTime / 1000.0);

    return 0;
}
