// CatBoost-MLX standalone CSV training tool.
// Reads a CSV file, quantizes features, and trains a GBDT model on Apple Silicon GPU.
//
// Usage: ./csv_train <file.csv> [options]
//   --iterations N         Number of boosting iterations (default: 100)
//   --depth D              Max tree depth (default: 6)
//   --lr RATE              Learning rate (default: 0.1)
//   --l2 LAMBDA            L2 regularization (default: 3.0)
//   --loss TYPE            Loss: rmse, logloss, multiclass, mae, quantile[:alpha], huber[:delta], poisson, tweedie[:p], mape, auto
//   --bins B               Max quantization bins per feature (default: 255)
//   --target-col N         0-based column index for target (default: last column)
//   --cat-features L       Comma-separated 0-based column indices for categorical features
//   --eval-fraction F      Fraction of data for validation (default: 0 = no split)
//   --early-stopping N     Stop after N iterations with no validation improvement (default: 0 = disabled)
//   --subsample F          Row subsampling fraction per iteration (default: 1.0)
//   --colsample-bytree F   Feature subsampling fraction per tree (default: 1.0)
//   --seed N               Random seed for subsampling (default: 42)
//   --nan-mode MODE        NaN handling: min (default), forbidden
//   --output PATH          Save trained model to JSON file
//   --feature-importance   Print gain-based feature importance after training
//   --cv N                 N-fold cross-validation (default: 0 = disabled)
//   --ctr                  Enable CTR target encoding for high-cardinality categoricals
//   --ctr-prior F          CTR prior (default: 0.5)
//   --max-onehot-size N    Max categories for OneHot; above uses CTR (default: 10)
//   --group-col N          0-based column index for group/query ID (ranking losses)
//   --weight-col N         0-based column index for sample weights (default: -1 = uniform)
//   --min-data-in-leaf N   Minimum documents per leaf (default: 1)
//   --monotone-constraints 0,1,-1,...  Per-feature monotone constraints (0=none, 1=inc, -1=dec)
//   --snapshot-path PATH   Save/restore training snapshot for resume (default: disabled)
//   --snapshot-interval N  Save snapshot every N iterations (default: 1)
//   --verbose              Print per-iteration loss
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
#include <unordered_map>
#include <unordered_set>
#include <random>

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
    bool IsOneHot = false;
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
    std::unordered_set<int> CatFeatureCols;  // 0-based column indices for categorical features
    // Early stopping
    float EvalFraction = 0.0f;         // fraction of data for validation (0 = disabled)
    ui32 EarlyStoppingPatience = 0;    // 0 = disabled
    // Subsampling
    float SubsampleRatio = 1.0f;       // row subsampling fraction per iteration
    float ColsampleByTree = 1.0f;      // feature subsampling fraction per tree
    ui32 RandomSeed = 42;
    // Bootstrap
    std::string BootstrapType = "no";  // "no", "bayesian", "bernoulli", "mvs"
    float BaggingTemperature = 1.0f;   // Bayesian bootstrap temperature
    float MvsReg = 0.0f;              // MVS regularization
    // Ranking / groups
    int GroupCol = -1;                 // -1 = no group column; 0-based column index for group/query ID
    // Sample weights
    int WeightCol = -1;                // -1 = no weight column; 0-based column index for sample weights
    // NaN handling
    std::string NanMode = "min";       // "min" = NaN → bin 0, "forbidden" = error on NaN
    // Model output
    std::string OutputModelPath;       // "" = no save
    bool ShowFeatureImportance = false; // print gain-based feature importance
    // Cross-validation
    ui32 CVFolds = 0;                  // 0 = disabled, N = N-fold cross-validation
    // CTR target encoding
    bool UseCtr = false;               // enable CTR for high-cardinality categoricals
    float CtrPrior = 0.5f;            // Bayesian smoothing prior
    ui32 MaxOneHotSize = 10;           // categoricals with > this many values use CTR instead of OneHot
    // Regularization
    ui32 MinDataInLeaf = 1;            // minimum documents per leaf (1 = no restriction)
    // Monotone constraints
    std::vector<int> MonotoneConstraints;  // per-feature: 0=none, 1=increasing, -1=decreasing
    // Snapshot save/resume
    std::string SnapshotPath;              // "" = disabled
    ui32 SnapshotInterval = 1;             // save snapshot every N iterations
    // External eval file
    std::string EvalFile;                  // "" = disabled; path to separate validation CSV
};

TConfig ParseArgs(int argc, char** argv) {
    TConfig config;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file.csv> [--iterations N] [--depth D] [--lr RATE] "
                "[--l2 LAMBDA] [--loss TYPE] [--bins B] [--target-col N] [--cat-features L] "
                "[--eval-fraction F] [--early-stopping N] [--subsample F] [--colsample-bytree F] "
                "[--seed N] [--nan-mode MODE] [--output PATH] [--feature-importance] "
                "[--cv N] [--ctr] [--ctr-prior F] [--max-onehot-size N] [--group-col N] "
                "[--weight-col N] [--min-data-in-leaf N] [--monotone-constraints L] "
                "[--snapshot-path PATH] [--snapshot-interval N] [--eval-file PATH] [--verbose]\n", argv[0]);
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
        else if (strcmp(argv[i], "--cat-features") == 0 && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                config.CatFeatureCols.insert(std::atoi(token.c_str()));
            }
        }
        else if (strcmp(argv[i], "--eval-fraction") == 0 && i + 1 < argc) config.EvalFraction = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--early-stopping") == 0 && i + 1 < argc) config.EarlyStoppingPatience = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--subsample") == 0 && i + 1 < argc) config.SubsampleRatio = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--colsample-bytree") == 0 && i + 1 < argc) config.ColsampleByTree = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) config.RandomSeed = std::atoll(argv[++i]);
        else if (strcmp(argv[i], "--bootstrap-type") == 0 && i + 1 < argc) config.BootstrapType = argv[++i];
        else if (strcmp(argv[i], "--bagging-temperature") == 0 && i + 1 < argc) config.BaggingTemperature = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--mvs-reg") == 0 && i + 1 < argc) config.MvsReg = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--nan-mode") == 0 && i + 1 < argc) config.NanMode = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) config.OutputModelPath = argv[++i];
        else if (strcmp(argv[i], "--feature-importance") == 0) config.ShowFeatureImportance = true;
        else if (strcmp(argv[i], "--cv") == 0 && i + 1 < argc) config.CVFolds = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctr") == 0) config.UseCtr = true;
        else if (strcmp(argv[i], "--ctr-prior") == 0 && i + 1 < argc) config.CtrPrior = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--max-onehot-size") == 0 && i + 1 < argc) config.MaxOneHotSize = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--group-col") == 0 && i + 1 < argc) config.GroupCol = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--weight-col") == 0 && i + 1 < argc) config.WeightCol = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-data-in-leaf") == 0 && i + 1 < argc) config.MinDataInLeaf = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--monotone-constraints") == 0 && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                config.MonotoneConstraints.push_back(std::atoi(token.c_str()));
            }
        }
        else if (strcmp(argv[i], "--snapshot-path") == 0 && i + 1 < argc) config.SnapshotPath = argv[++i];
        else if (strcmp(argv[i], "--snapshot-interval") == 0 && i + 1 < argc) config.SnapshotInterval = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--eval-file") == 0 && i + 1 < argc) config.EvalFile = argv[++i];
        else if (strcmp(argv[i], "--verbose") == 0) config.Verbose = true;
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); exit(1); }
    }
    return config;
}

// ============================================================================
// CSV loading
// ============================================================================

struct TDataset {
    std::vector<std::vector<float>> Features;  // [numFeatures][numDocs] — float values (or hash indices for categorical)
    std::vector<float> Targets;                 // [numDocs]
    ui32 NumDocs = 0;
    ui32 NumFeatures = 0;
    std::vector<std::string> FeatureNames;
    std::vector<bool> IsCategorical;            // [numFeatures] — true for categorical columns
    std::vector<std::unordered_map<std::string, uint32_t>> CatHashMaps;  // [numFeatures] — string → bin index (empty for numeric)
    std::vector<bool> HasNaN;                   // [numFeatures] — true if any value is NaN
    // Group/query data (for ranking losses)
    std::vector<ui32> GroupIds;                  // [numDocs] — integer group ID per doc
    std::vector<ui32> GroupOffsets;              // [numGroups + 1] — prefix sum; group g spans [GroupOffsets[g], GroupOffsets[g+1])
    ui32 NumGroups = 0;
    // Sample weights
    std::vector<float> Weights;                  // [numDocs] — per-sample weights (empty = uniform)
};

bool IsNumber(const std::string& s) {
    if (s.empty()) return false;
    // NaN/missing value markers are "numeric-ish" — they'll become NaN during parsing
    if (s == "NaN" || s == "nan" || s == "NA" || s == "na" || s == "N/A" || s == "?") return true;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

bool IsNaNString(const std::string& s) {
    return s.empty() || s == "NaN" || s == "nan" || s == "NA" || s == "na" || s == "N/A" || s == "?";
}

TDataset LoadCSV(const std::string& path, int targetCol, const std::unordered_set<int>& catFeatureCols,
                  const std::string& nanMode, int groupCol = -1, int weightCol = -1) {
    TDataset ds;
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file: %s\n", path.c_str());
        exit(1);
    }

    std::string line;
    std::vector<std::vector<std::string>> rawRows;  // store all cells as strings first
    bool hasHeader = false;
    int numCols = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) {
            size_t start = cell.find_first_not_of(" \t");
            size_t end = cell.find_last_not_of(" \t");
            if (start != std::string::npos) cell = cell.substr(start, end - start + 1);
            else cell = "";
            cells.push_back(cell);
        }

        if (numCols == 0) {
            numCols = cells.size();
            if (!cells.empty() && !IsNumber(cells[0])) {
                hasHeader = true;
                ds.FeatureNames = cells;
                continue;
            }
        }

        if (static_cast<int>(cells.size()) != numCols) continue;
        rawRows.push_back(cells);
    }

    if (rawRows.empty()) {
        fprintf(stderr, "Error: No valid data rows in %s\n", path.c_str());
        exit(1);
    }

    ds.NumDocs = rawRows.size();
    int tgtCol = (targetCol >= 0) ? targetCol : (numCols - 1);
    if (tgtCol < 0 || tgtCol >= numCols) {
        fprintf(stderr, "Error: Invalid target column %d (data has %d columns)\n", tgtCol, numCols);
        exit(1);
    }

    // Determine which feature columns are categorical
    // Exclude target, group, and weight columns from features
    ui32 excludedCols = 1 + (groupCol >= 0 ? 1 : 0) + (weightCol >= 0 ? 1 : 0);
    ds.NumFeatures = numCols - excludedCols;
    ds.IsCategorical.resize(ds.NumFeatures, false);

    // Fix feature names: exclude the target column, group column, and weight column
    if (hasHeader && !ds.FeatureNames.empty()) {
        std::vector<std::string> filteredNames;
        for (int c = 0; c < numCols; ++c) {
            if (c == tgtCol) continue;
            if (c == groupCol) continue;
            if (c == weightCol) continue;
            if (c < static_cast<int>(ds.FeatureNames.size()))
                filteredNames.push_back(ds.FeatureNames[c]);
        }
        ds.FeatureNames = std::move(filteredNames);
    }

    // Map from original column index (excluding target, group, and weight) to feature index
    std::vector<int> colToFeatIdx(numCols, -1);
    {
        ui32 fIdx = 0;
        for (int c = 0; c < numCols; ++c) {
            if (c == tgtCol) continue;
            if (c == groupCol) continue;
            if (c == weightCol) continue;
            colToFeatIdx[c] = fIdx;

            // Check if this column was explicitly marked as categorical
            if (catFeatureCols.count(c)) {
                ds.IsCategorical[fIdx] = true;
            }
            fIdx++;
        }
    }

    // Auto-detect: if not explicitly specified, check if any cell in a non-target column is non-numeric
    // Empty cells and NaN markers (NaN, NA, na, N/A, ?) are treated as missing values, not categorical
    if (catFeatureCols.empty()) {
        for (int c = 0; c < numCols; ++c) {
            if (c == tgtCol) continue;
            if (c == groupCol) continue;
            if (c == weightCol) continue;
            int fIdx = colToFeatIdx[c];
            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                const std::string& val = rawRows[d][c];
                if (!IsNumber(val) && !IsNaNString(val)) {
                    ds.IsCategorical[fIdx] = true;
                    break;
                }
            }
        }
    }

    // Build hash maps for categorical features, parse numeric features
    ds.Features.resize(ds.NumFeatures);
    ds.CatHashMaps.resize(ds.NumFeatures);
    ds.HasNaN.resize(ds.NumFeatures, false);
    for (ui32 f = 0; f < ds.NumFeatures; ++f) {
        ds.Features[f].resize(ds.NumDocs);
    }
    ds.Targets.resize(ds.NumDocs);

    // Parse group column strings → integer group IDs
    std::unordered_map<std::string, uint32_t> groupStringToId;
    if (groupCol >= 0) {
        ds.GroupIds.resize(ds.NumDocs);
        for (ui32 d = 0; d < ds.NumDocs; ++d) {
            const std::string& val = rawRows[d][groupCol];
            auto it = groupStringToId.find(val);
            if (it == groupStringToId.end()) {
                uint32_t newId = groupStringToId.size();
                groupStringToId[val] = newId;
                ds.GroupIds[d] = newId;
            } else {
                ds.GroupIds[d] = it->second;
            }
        }
    }

    // Parse weight column if present
    if (weightCol >= 0) {
        ds.Weights.resize(ds.NumDocs);
        for (ui32 d = 0; d < ds.NumDocs; ++d) {
            try {
                ds.Weights[d] = std::stof(rawRows[d][weightCol]);
            } catch (...) {
                fprintf(stderr, "Error: Non-numeric weight at row %u: '%s'\n", d, rawRows[d][weightCol].c_str());
                exit(1);
            }
        }
    }

    for (ui32 d = 0; d < ds.NumDocs; ++d) {
        // Parse target (must be numeric)
        try {
            ds.Targets[d] = std::stof(rawRows[d][tgtCol]);
        } catch (...) {
            fprintf(stderr, "Error: Non-numeric target at row %u: '%s'\n", d, rawRows[d][tgtCol].c_str());
            exit(1);
        }

        ui32 fIdx = 0;
        for (int c = 0; c < numCols; ++c) {
            if (c == tgtCol) continue;
            if (c == groupCol) continue;
            if (c == weightCol) continue;
            if (ds.IsCategorical[fIdx]) {
                // Categorical: hash string → sequential integer
                const std::string& val = rawRows[d][c];
                auto& hashMap = ds.CatHashMaps[fIdx];
                auto it = hashMap.find(val);
                if (it == hashMap.end()) {
                    uint32_t newIdx = hashMap.size();
                    hashMap[val] = newIdx;
                    ds.Features[fIdx][d] = static_cast<float>(newIdx);
                } else {
                    ds.Features[fIdx][d] = static_cast<float>(it->second);
                }
            } else {
                // Numeric — handle NaN/missing values
                const std::string& val = rawRows[d][c];
                if (IsNaNString(val)) {
                    if (nanMode == "forbidden") {
                        fprintf(stderr, "Error: NaN/missing value at row %u, col %d: '%s' (--nan-mode=forbidden)\n",
                                d, c, val.c_str());
                        exit(1);
                    }
                    ds.Features[fIdx][d] = std::numeric_limits<float>::quiet_NaN();
                    ds.HasNaN[fIdx] = true;
                } else {
                    try {
                        ds.Features[fIdx][d] = std::stof(val);
                        if (std::isnan(ds.Features[fIdx][d])) {
                            ds.HasNaN[fIdx] = true;
                        }
                    } catch (...) {
                        if (nanMode == "forbidden") {
                            fprintf(stderr, "Error: Non-numeric value at row %u, col %d: '%s'\n", d, c, val.c_str());
                            exit(1);
                        }
                        ds.Features[fIdx][d] = std::numeric_limits<float>::quiet_NaN();
                        ds.HasNaN[fIdx] = true;
                    }
                }
            }
            fIdx++;
        }
    }

    // Sort data by group ID and build GroupOffsets
    if (groupCol >= 0 && !ds.GroupIds.empty()) {
        // Create a permutation that sorts by group ID (stable sort preserves order within groups)
        std::vector<ui32> perm(ds.NumDocs);
        std::iota(perm.begin(), perm.end(), 0);
        std::stable_sort(perm.begin(), perm.end(),
            [&](ui32 a, ui32 b) { return ds.GroupIds[a] < ds.GroupIds[b]; });

        // Apply permutation to all data
        auto permute = [&](auto& vec) {
            auto tmp = vec;
            for (ui32 i = 0; i < ds.NumDocs; ++i) vec[i] = tmp[perm[i]];
        };
        permute(ds.Targets);
        permute(ds.GroupIds);
        if (!ds.Weights.empty()) permute(ds.Weights);
        for (ui32 f = 0; f < ds.NumFeatures; ++f) {
            permute(ds.Features[f]);
        }

        // Build GroupOffsets prefix sum
        ds.NumGroups = 0;
        ds.GroupOffsets.push_back(0);
        for (ui32 d = 1; d < ds.NumDocs; ++d) {
            if (ds.GroupIds[d] != ds.GroupIds[d - 1]) {
                ds.GroupOffsets.push_back(d);
                ds.NumGroups++;
            }
        }
        ds.GroupOffsets.push_back(ds.NumDocs);
        ds.NumGroups++;
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
        if (ds.IsCategorical[f]) {
            // Categorical: no borders, bin index = hash map index directly
            q.Borders[f].clear();
            q.BinnedFeatures[f].resize(ds.NumDocs);
            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                q.BinnedFeatures[f][d] = static_cast<uint8_t>(ds.Features[f][d]);
            }
            continue;
        }

        // Numeric: equal-frequency quantization
        // Filter out NaN values before border computation
        std::vector<float> sorted;
        sorted.reserve(ds.NumDocs);
        for (ui32 d = 0; d < ds.NumDocs; ++d) {
            if (!std::isnan(ds.Features[f][d])) {
                sorted.push_back(ds.Features[f][d]);
            }
        }
        if (sorted.empty()) {
            // All NaN — single NaN bin
            q.BinnedFeatures[f].resize(ds.NumDocs, 0);
            continue;
        }
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

        // Quantize: NaN → bin 0, real values → bins 1..numBorders+1
        bool hasNaN = ds.HasNaN[f];
        ui32 binOffset = hasNaN ? 1 : 0;
        q.BinnedFeatures[f].resize(ds.NumDocs);
        for (ui32 d = 0; d < ds.NumDocs; ++d) {
            if (std::isnan(ds.Features[f][d])) {
                q.BinnedFeatures[f][d] = 0;  // NaN bin
            } else {
                auto it = std::upper_bound(q.Borders[f].begin(), q.Borders[f].end(), ds.Features[f][d]);
                q.BinnedFeatures[f][d] = static_cast<uint8_t>((it - q.Borders[f].begin()) + binOffset);
            }
        }
    }

    return q;
}

// Quantize a new dataset using pre-computed borders from training data
TQuantization QuantizeWithBorders(const TDataset& ds, const TQuantization& trainQuant,
                                  const TDataset& trainDs) {
    TQuantization q;
    q.Borders = trainQuant.Borders;
    q.BinnedFeatures.resize(ds.NumFeatures);

    for (ui32 f = 0; f < ds.NumFeatures; ++f) {
        q.BinnedFeatures[f].resize(ds.NumDocs);
        if (ds.IsCategorical[f]) {
            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                q.BinnedFeatures[f][d] = static_cast<uint8_t>(ds.Features[f][d]);
            }
        } else {
            bool hasNaN = trainDs.HasNaN[f];  // use training NaN status for consistent bin offsets
            ui32 binOffset = hasNaN ? 1 : 0;
            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                if (std::isnan(ds.Features[f][d])) {
                    q.BinnedFeatures[f][d] = 0;
                } else {
                    auto it = std::upper_bound(q.Borders[f].begin(), q.Borders[f].end(), ds.Features[f][d]);
                    q.BinnedFeatures[f][d] = static_cast<uint8_t>((it - q.Borders[f].begin()) + binOffset);
                }
            }
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

TPackedData PackFeatures(const TQuantization& q, const TDataset& ds) {
    const ui32 numDocs = ds.NumDocs;
    const ui32 numFeatures = ds.NumFeatures;
    TPackedData packed;
    packed.NumUi32PerDoc = (numFeatures + 3) / 4;
    packed.Data.resize(numDocs * packed.NumUi32PerDoc, 0);
    packed.TotalBinFeatures = 0;

    for (ui32 f = 0; f < numFeatures; ++f) {
        ui32 wordIdx = f / 4;
        ui32 posInWord = f % 4;
        ui32 shift = (3 - posInWord) * 8;
        ui32 mask = 0xFF << shift;

        // For categorical features, folds = number of unique categories
        // For numeric features, folds = number of borders (+ 1 if feature has NaN for the NaN bin)
        ui32 folds;
        if (ds.IsCategorical[f]) {
            folds = static_cast<ui32>(ds.CatHashMaps[f].size());
        } else {
            folds = static_cast<ui32>(q.Borders[f].size());
            if (ds.HasNaN[f]) folds += 1;  // extra bin 0 for NaN
        }

        TCFeature feat;
        feat.Offset = wordIdx;
        feat.Mask = mask;
        feat.Shift = shift;
        feat.FirstFoldIndex = packed.TotalBinFeatures;
        feat.Folds = folds;
        feat.OneHotFeature = ds.IsCategorical[f];
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
    ui32 numPartitions,
    const std::vector<bool>& featureMask = {},  // optional: if non-empty, skip features where mask[f]=false
    ui32 minDataInLeaf = 1,                     // minimum docs per child (1 = no restriction)
    const std::vector<std::vector<ui32>>& countHist = {},  // [numPartitions][totalBinFeatures] doc counts per bin
    const std::vector<ui32>& partDocCounts = {},            // [numPartitions] total doc count per partition
    const std::vector<int>& monotoneConstraints = {}       // per-feature: 0=none, 1=inc, -1=dec
) {
    TBestSplitProperties bestSplit;
    const ui32 K = perDimHist.size();
    float bestGain = -std::numeric_limits<float>::infinity();

    // The Metal histogram kernel produces per-bin sums with a +1 offset:
    //   hist[firstFold + b] = sum of docs where featureValue == b+1
    //
    // For OneHot features: each bin represents one category.
    //   Split "go right if value == bin": sumRight = hist[bin], sumLeft = total - sumRight
    //
    // For ordinal features: suffix-sum to compute left/right.
    //   Split threshold b (value > b → right):
    //     sumRight = sum(hist[b..folds-1]) = docs with value ∈ {b+1,...,folds}
    //     sumLeft = total - sumRight

    for (ui32 featIdx = 0; featIdx < features.size(); ++featIdx) {
        // Feature subsampling: skip features not in the selected set
        if (!featureMask.empty() && !featureMask[featIdx]) continue;
        const auto& feat = features[featIdx];

        for (ui32 bin = 0; bin < feat.Folds; ++bin) {
            float totalGain = 0.0f;
            bool violatesConstraint = false;

            // Min-data-in-leaf check: verify doc counts per partition
            if (minDataInLeaf > 1 && !countHist.empty()) {
                bool anyPartitionViolates = false;
                for (ui32 p = 0; p < numPartitions; ++p) {
                    ui32 countRight = 0;
                    if (feat.OneHotFeature) {
                        countRight = countHist[p][feat.FirstFoldIndex + bin];
                    } else {
                        for (ui32 b = bin; b < feat.Folds; ++b)
                            countRight += countHist[p][feat.FirstFoldIndex + b];
                    }
                    ui32 countLeft = partDocCounts[p] - countRight;
                    if (countLeft < minDataInLeaf || countRight < minDataInLeaf) {
                        anyPartitionViolates = true;
                        break;
                    }
                }
                if (anyPartitionViolates) continue;
            }

            for (ui32 p = 0; p < numPartitions; ++p) {
                // Monotone constraint check: use first dimension (k=0) for leaf value direction
                if (!monotoneConstraints.empty() && featIdx < monotoneConstraints.size()
                    && monotoneConstraints[featIdx] != 0 && !feat.OneHotFeature) {
                    const float* histData0 = perDimHist[0].data() + p * 2 * totalBinFeatures;
                    float tSum = static_cast<float>(perDimPartStats[0][p].Sum);
                    float tWeight = static_cast<float>(perDimPartStats[0][p].Weight);
                    float sR, wR;
                    sR = 0.0f; wR = 0.0f;
                    for (ui32 b = bin; b < feat.Folds; ++b) {
                        sR += histData0[feat.FirstFoldIndex + b];
                        wR += histData0[totalBinFeatures + feat.FirstFoldIndex + b];
                    }
                    float sL = tSum - sR;
                    float wL = tWeight - wR;
                    if (wL > 1e-15f && wR > 1e-15f) {
                        // Leaf values: v = -lr * G/(H+λ). For +1 constraint, v_right >= v_left.
                        // Since leaf = -lr * G/(H+λ), right >= left iff G_L/(H_L+λ) >= G_R/(H_R+λ)
                        float vL = sL / (wL + l2RegLambda);
                        float vR = sR / (wR + l2RegLambda);
                        if (monotoneConstraints[featIdx] == 1 && vL < vR) {
                            violatesConstraint = true;
                            break;
                        }
                        if (monotoneConstraints[featIdx] == -1 && vL > vR) {
                            violatesConstraint = true;
                            break;
                        }
                    }
                }

                for (ui32 k = 0; k < K; ++k) {
                    const float* histData = perDimHist[k].data() + p * 2 * totalBinFeatures;

                    float totalSum = static_cast<float>(perDimPartStats[k][p].Sum);
                    float totalWeight = static_cast<float>(perDimPartStats[k][p].Weight);

                    float sumRight, weightRight;
                    if (feat.OneHotFeature) {
                        // OneHot: hist[bin] = docs where featureValue == bin+1
                        sumRight = histData[feat.FirstFoldIndex + bin];
                        weightRight = histData[totalBinFeatures + feat.FirstFoldIndex + bin];
                    } else {
                        // Ordinal: suffix sum over hist[bin..folds-1]
                        sumRight = 0.0f;
                        weightRight = 0.0f;
                        for (ui32 b = bin; b < feat.Folds; ++b) {
                            sumRight += histData[feat.FirstFoldIndex + b];
                            weightRight += histData[totalBinFeatures + feat.FirstFoldIndex + b];
                        }
                    }

                    float sumLeft = totalSum - sumRight;
                    float weightLeft = totalWeight - weightRight;

                    // Skip partitions where one side is empty — zero gain contribution
                    if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;

                    totalGain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda)
                               + (sumRight * sumRight) / (weightRight + l2RegLambda)
                               - (totalSum * totalSum) / (totalWeight + l2RegLambda);
                }
            }

            if (violatesConstraint) continue;

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
// Loss function helpers: parse loss type with optional parameter
// ============================================================================

struct TLossConfig {
    std::string Type;
    float Param = 0.0f;  // alpha for quantile, delta for huber
};

TLossConfig ParseLossType(const std::string& lossStr) {
    TLossConfig lc;
    auto colonPos = lossStr.find(':');
    if (colonPos != std::string::npos) {
        lc.Type = lossStr.substr(0, colonPos);
        lc.Param = std::stof(lossStr.substr(colonPos + 1));
    } else {
        lc.Type = lossStr;
        if (lc.Type == "quantile") lc.Param = 0.5f;
        if (lc.Type == "huber") lc.Param = 1.0f;
        if (lc.Type == "tweedie") lc.Param = 1.5f;  // variance power p
    }
    return lc;
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
// Compute loss value for a given cursor, targets, loss type
// ============================================================================

float ComputeLossValue(
    const mx::array& cursor, const mx::array& targetsArr,
    const std::string& lossType, float lossParam, ui32 approxDim, ui32 numDocs
) {
    if (lossType == "rmse") {
        auto diff = mx::subtract(mx::reshape(cursor, {static_cast<int>(numDocs)}), targetsArr);
        auto loss = mx::sqrt(mx::mean(mx::multiply(diff, diff)));
        mx::eval(loss);
        return loss.item<float>();
    } else if (lossType == "mae") {
        auto diff = mx::subtract(mx::reshape(cursor, {static_cast<int>(numDocs)}), targetsArr);
        auto loss = mx::mean(mx::abs(diff));
        mx::eval(loss);
        return loss.item<float>();
    } else if (lossType == "quantile") {
        float alpha = lossParam;
        auto diff = mx::subtract(targetsArr, mx::reshape(cursor, {static_cast<int>(numDocs)}));
        auto isPos = mx::greater(diff, mx::array(0.0f));
        auto loss = mx::where(isPos,
            mx::multiply(mx::array(alpha), diff),
            mx::multiply(mx::array(alpha - 1.0f), diff));
        auto result = mx::mean(loss);
        mx::eval(result);
        return result.item<float>();
    } else if (lossType == "huber") {
        float delta = lossParam;
        auto diff = mx::subtract(mx::reshape(cursor, {static_cast<int>(numDocs)}), targetsArr);
        auto absDiff = mx::abs(diff);
        auto isSmall = mx::less_equal(absDiff, mx::array(delta));
        auto loss = mx::where(isSmall,
            mx::multiply(mx::array(0.5f), mx::multiply(diff, diff)),
            mx::subtract(mx::multiply(mx::array(delta), absDiff), mx::array(0.5f * delta * delta)));
        auto result = mx::mean(loss);
        mx::eval(result);
        return result.item<float>();
    } else if (lossType == "logloss") {
        auto sig = mx::sigmoid(mx::reshape(cursor, {static_cast<int>(numDocs)}));
        auto eps = mx::array(1e-15f);
        auto loss = mx::negative(mx::mean(mx::add(
            mx::multiply(targetsArr, mx::log(mx::add(sig, eps))),
            mx::multiply(mx::subtract(mx::array(1.0f), targetsArr),
                         mx::log(mx::add(mx::subtract(mx::array(1.0f), sig), eps)))
        )));
        mx::eval(loss);
        return loss.item<float>();
    } else if (lossType == "poisson") {
        auto flatCursor = mx::reshape(cursor, {static_cast<int>(numDocs)});
        auto expPred = mx::exp(flatCursor);
        auto loss = mx::mean(mx::subtract(expPred, mx::multiply(targetsArr, flatCursor)));
        mx::eval(loss);
        return loss.item<float>();
    } else if (lossType == "tweedie") {
        float p = lossParam;
        auto flatCursor = mx::reshape(cursor, {static_cast<int>(numDocs)});
        auto term1 = mx::divide(
            mx::multiply(mx::negative(targetsArr), mx::exp(mx::multiply(flatCursor, mx::array(1.0f - p)))),
            mx::array(1.0f - p));
        auto term2 = mx::divide(
            mx::exp(mx::multiply(flatCursor, mx::array(2.0f - p))),
            mx::array(2.0f - p));
        auto loss = mx::mean(mx::add(term1, term2));
        mx::eval(loss);
        return loss.item<float>();
    } else if (lossType == "mape") {
        auto flatCursor = mx::reshape(cursor, {static_cast<int>(numDocs)});
        auto absTarget = mx::maximum(mx::abs(targetsArr), mx::array(1e-6f));
        auto loss = mx::mean(mx::divide(mx::abs(mx::subtract(flatCursor, targetsArr)), absTarget));
        mx::eval(loss);
        return loss.item<float>();
    } else {
        // multiclass
        auto sm = ComputeSoftmax(cursor, approxDim, numDocs);
        auto targetInt = mx::astype(targetsArr, mx::int32);
        auto probTarget = mx::zeros({static_cast<int>(numDocs)}, mx::float32);
        for (ui32 k = 0; k < approxDim; ++k) {
            auto isClass = mx::astype(mx::equal(targetInt, mx::array(static_cast<int>(k))), mx::float32);
            auto probK = mx::reshape(
                mx::slice(sm.Probs, {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)}),
                {static_cast<int>(numDocs)});
            probTarget = mx::add(probTarget, mx::multiply(isClass, probK));
        }
        auto isLast = mx::astype(mx::equal(targetInt, mx::array(static_cast<int>(approxDim))), mx::float32);
        auto probImp = mx::subtract(mx::array(1.0f), mx::sum(sm.Probs, 0));
        probTarget = mx::add(probTarget, mx::multiply(isLast, probImp));
        auto loss = mx::negative(mx::mean(mx::log(mx::add(probTarget, mx::array(1e-15f)))));
        mx::eval(loss);
        return loss.item<float>();
    }
}

// ============================================================================
// Ranking: pair generation and gradient computation
// ============================================================================

struct TPair {
    ui32 Winner;  // doc index with higher relevance
    ui32 Loser;   // doc index with lower relevance
    float Weight;
};

// Generate all (winner, loser) pairs within each group for PairLogit
std::vector<TPair> GeneratePairLogitPairs(
    const std::vector<float>& targets,
    const std::vector<ui32>& groupOffsets,
    ui32 numGroups
) {
    std::vector<TPair> pairs;
    for (ui32 g = 0; g < numGroups; ++g) {
        ui32 begin = groupOffsets[g];
        ui32 end = groupOffsets[g + 1];
        for (ui32 i = begin; i < end; ++i) {
            for (ui32 j = begin; j < end; ++j) {
                if (targets[i] > targets[j]) {
                    pairs.push_back({i, j, 1.0f});
                }
            }
        }
    }
    return pairs;
}

// Generate YetiRank pairs: random permutation within each group,
// then adjacent-position pairs weighted by position and relevance difference
std::vector<TPair> GenerateYetiRankPairs(
    const std::vector<float>& targets,
    const std::vector<ui32>& groupOffsets,
    ui32 numGroups,
    std::mt19937& rng
) {
    std::vector<TPair> pairs;
    for (ui32 g = 0; g < numGroups; ++g) {
        ui32 begin = groupOffsets[g];
        ui32 end = groupOffsets[g + 1];
        ui32 groupSize = end - begin;
        if (groupSize < 2) continue;

        // Random permutation of doc indices within group
        std::vector<ui32> perm(groupSize);
        std::iota(perm.begin(), perm.end(), begin);
        std::shuffle(perm.begin(), perm.end(), rng);

        // Adjacent pairs in the random permutation
        for (ui32 pos = 1; pos < groupSize; ++pos) {
            ui32 i = perm[pos - 1];
            ui32 j = perm[pos];
            float relevDiff = std::fabs(targets[i] - targets[j]);
            if (relevDiff < 1e-8f) continue;
            float weight = relevDiff / std::log2(2.0f + pos);
            if (targets[i] > targets[j]) {
                pairs.push_back({i, j, weight});
            } else {
                pairs.push_back({j, i, weight});
            }
        }
    }
    return pairs;
}

// Scatter pairwise gradients to per-document gradient/hessian arrays
void ScatterPairwiseGradients(
    const std::vector<TPair>& pairs,
    const float* preds,  // raw predictions [numDocs]
    ui32 numDocs,
    std::vector<float>& grads,
    std::vector<float>& hess
) {
    grads.assign(numDocs, 0.0f);
    hess.assign(numDocs, 0.0f);

    for (const auto& pair : pairs) {
        float diff = preds[pair.Winner] - preds[pair.Loser];
        float p = 1.0f / (1.0f + std::exp(-diff));  // sigmoid(diff)
        float w = pair.Weight;

        // gradient: push winner score up, loser score down
        grads[pair.Winner] += w * (p - 1.0f);
        grads[pair.Loser]  += w * (1.0f - p);

        // hessian: p*(1-p) for both
        float h = w * p * (1.0f - p);
        hess[pair.Winner] += h;
        hess[pair.Loser]  += h;
    }

    // Floor hessians to avoid zero
    for (ui32 d = 0; d < numDocs; ++d) {
        hess[d] = std::max(hess[d], 1e-6f);
    }
}

// PairLogit loss: sum over pairs: log(1 + exp(-(pred[winner] - pred[loser])))
float ComputePairLogitLoss(
    const std::vector<TPair>& pairs,
    const float* preds,
    ui32 /*numDocs*/
) {
    if (pairs.empty()) return 0.0f;
    double totalLoss = 0.0;
    for (const auto& pair : pairs) {
        float diff = preds[pair.Winner] - preds[pair.Loser];
        totalLoss += pair.Weight * std::log(1.0f + std::exp(-diff));
    }
    return static_cast<float>(totalLoss / pairs.size());
}

// NDCG metric: per-group DCG/IDCG averaged across groups
float ComputeNDCG(
    const std::vector<float>& targets,
    const float* preds,
    const std::vector<ui32>& groupOffsets,
    ui32 numGroups
) {
    if (numGroups == 0) return 0.0f;
    double totalNDCG = 0.0;
    ui32 validGroups = 0;

    for (ui32 g = 0; g < numGroups; ++g) {
        ui32 begin = groupOffsets[g];
        ui32 end = groupOffsets[g + 1];
        ui32 groupSize = end - begin;
        if (groupSize < 2) continue;

        // Sort docs by predicted score (descending)
        std::vector<ui32> sortedIdx(groupSize);
        std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
        std::sort(sortedIdx.begin(), sortedIdx.end(),
            [&](ui32 a, ui32 b) { return preds[begin + a] > preds[begin + b]; });

        // Compute DCG
        double dcg = 0.0;
        for (ui32 rank = 0; rank < groupSize; ++rank) {
            float relev = targets[begin + sortedIdx[rank]];
            dcg += (std::pow(2.0, relev) - 1.0) / std::log2(2.0 + rank);
        }

        // Compute IDCG (sort by true relevance descending)
        std::vector<float> sortedRelev(groupSize);
        for (ui32 i = 0; i < groupSize; ++i) sortedRelev[i] = targets[begin + i];
        std::sort(sortedRelev.begin(), sortedRelev.end(), std::greater<float>());

        double idcg = 0.0;
        for (ui32 rank = 0; rank < groupSize; ++rank) {
            idcg += (std::pow(2.0, sortedRelev[rank]) - 1.0) / std::log2(2.0 + rank);
        }

        if (idcg > 1e-10) {
            totalNDCG += dcg / idcg;
            validGroups++;
        }
    }

    return (validGroups > 0) ? static_cast<float>(totalNDCG / validGroups) : 0.0f;
}

// ============================================================================
// Apply tree splits to compute leaf indices for any compressed data
// ============================================================================

mx::array ComputeLeafIndices(
    const mx::array& compressedData,
    const std::vector<TObliviousSplitLevel>& splits,
    ui32 numDocs
) {
    auto leafIndices = mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
    for (ui32 level = 0; level < splits.size(); ++level) {
        const auto& split = splits[level];
        auto column = mx::slice(compressedData,
            {0, static_cast<int>(split.FeatureColumnIdx)},
            {static_cast<int>(numDocs), static_cast<int>(split.FeatureColumnIdx + 1)});
        column = mx::reshape(column, {static_cast<int>(numDocs)});
        auto featureValues = mx::bitwise_and(
            mx::right_shift(column, mx::array(static_cast<uint32_t>(split.Shift), mx::uint32)),
            mx::array(static_cast<uint32_t>(split.Mask), mx::uint32));
        auto goRight = split.IsOneHot
            ? mx::equal(featureValues, mx::array(static_cast<uint32_t>(split.BinThreshold), mx::uint32))
            : mx::greater(featureValues, mx::array(static_cast<uint32_t>(split.BinThreshold), mx::uint32));
        auto bits = mx::left_shift(mx::astype(goRight, mx::uint32), mx::array(static_cast<uint32_t>(level), mx::uint32));
        leafIndices = mx::bitwise_or(leafIndices, bits);
    }
    mx::eval(leafIndices);
    return leafIndices;
}

// ============================================================================
// CTR (target encoding) for high-cardinality categorical features
// ============================================================================

struct TCtrFeature {
    ui32 OrigFeatureIdx;               // index in original ds.Features
    std::string Name;                  // e.g. "city_ctr" or "city_ctr_class1"
    std::vector<float> Values;         // [numDocs] — ordered CTR values per doc
    // Final statistics for prediction (computed from ALL training data)
    std::unordered_map<uint32_t, float> FinalCtrValues;  // catBinIdx → final CTR
    float DefaultCtr;                  // CTR for unknown categories = prior / 1
    float Prior;                       // the prior used
    ui32 ClassIdx;                     // which class this CTR is for (0 for binary/regression)
    std::unordered_map<std::string, uint32_t> OrigCatHashMap;  // original string → bin mapping for prediction
};

// Compute ordered (online) CTR features.
// For each high-cardinality categorical feature, replaces it in the dataset with
// one or more numeric CTR features. Uses ordered computation to prevent target leakage.
std::vector<TCtrFeature> ComputeCtrFeatures(
    TDataset& ds,
    const std::string& lossType,
    ui32 numClasses,
    float prior,
    ui32 maxOneHotSize,
    ui32 randomSeed
) {
    std::vector<TCtrFeature> ctrFeatures;
    std::mt19937 rng(randomSeed + 12345);  // separate seed from subsampling

    // Create random permutation for ordered CTR
    std::vector<ui32> perm(ds.NumDocs);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    // Inverse permutation: originalIdx → positionInPerm
    std::vector<ui32> invPerm(ds.NumDocs);
    for (ui32 i = 0; i < ds.NumDocs; ++i) {
        invPerm[perm[i]] = i;
    }

    for (ui32 f = 0; f < ds.NumFeatures; ++f) {
        if (!ds.IsCategorical[f]) continue;
        ui32 numCategories = ds.CatHashMaps[f].size();
        if (numCategories <= maxOneHotSize) continue;  // keep OneHot for low-cardinality

        std::string baseName = (f < ds.FeatureNames.size() && !ds.FeatureNames[f].empty())
            ? ds.FeatureNames[f] : ("f" + std::to_string(f));

        if (lossType == "multiclass" && numClasses > 2) {
            // One CTR feature per class
            for (ui32 cls = 0; cls < numClasses; ++cls) {
                TCtrFeature ctr;
                ctr.OrigFeatureIdx = f;
                ctr.Name = baseName + "_ctr_class" + std::to_string(cls);
                ctr.Values.resize(ds.NumDocs);
                ctr.Prior = prior;
                ctr.ClassIdx = cls;
                ctr.DefaultCtr = prior / 1.0f;

                // Ordered CTR: process in permutation order
                // Per-category accumulators: countOfClass, totalCount
                std::unordered_map<uint32_t, float> catClassCount;  // sum of (target == cls) for docs before
                std::unordered_map<uint32_t, float> catTotalCount;  // total docs before

                for (ui32 i = 0; i < ds.NumDocs; ++i) {
                    ui32 doc = perm[i];
                    uint32_t catBin = static_cast<uint32_t>(ds.Features[f][doc]);

                    // CTR for this doc uses only stats from docs before it in permutation
                    float countBefore = catClassCount[catBin];
                    float totalBefore = catTotalCount[catBin];
                    ctr.Values[doc] = (countBefore + prior) / (totalBefore + 1.0f);

                    // Update accumulators with this doc's target
                    catTotalCount[catBin] += 1.0f;
                    if (static_cast<ui32>(ds.Targets[doc]) == cls) {
                        catClassCount[catBin] += 1.0f;
                    }
                }

                // Final CTR values (using ALL data) for prediction
                for (const auto& [catStr, catBin] : ds.CatHashMaps[f]) {
                    float cc = catClassCount[catBin];
                    float tc = catTotalCount[catBin];
                    ctr.FinalCtrValues[catBin] = (cc + prior) / (tc + 1.0f);
                }

                ctrFeatures.push_back(std::move(ctr));
            }
        } else {
            // Binary classification or regression: one CTR per feature
            TCtrFeature ctr;
            ctr.OrigFeatureIdx = f;
            ctr.Name = baseName + "_ctr";
            ctr.Values.resize(ds.NumDocs);
            ctr.Prior = prior;
            ctr.ClassIdx = 0;
            ctr.DefaultCtr = prior / 1.0f;

            // For logloss: countInClass = count of target==1
            // For regression: countInClass = sum of target values
            bool isBinaryClass = (lossType == "logloss");

            std::unordered_map<uint32_t, float> catSum;    // sum of targets (or positive count)
            std::unordered_map<uint32_t, float> catCount;  // total count

            for (ui32 i = 0; i < ds.NumDocs; ++i) {
                ui32 doc = perm[i];
                uint32_t catBin = static_cast<uint32_t>(ds.Features[f][doc]);

                float sumBefore = catSum[catBin];
                float countBefore = catCount[catBin];
                ctr.Values[doc] = (sumBefore + prior) / (countBefore + 1.0f);

                catCount[catBin] += 1.0f;
                if (isBinaryClass) {
                    catSum[catBin] += (ds.Targets[doc] > 0.5f) ? 1.0f : 0.0f;
                } else {
                    catSum[catBin] += ds.Targets[doc];
                }
            }

            // Final CTR values for prediction
            for (const auto& [catStr, catBin] : ds.CatHashMaps[f]) {
                float s = catSum[catBin];
                float c = catCount[catBin];
                ctr.FinalCtrValues[catBin] = (s + prior) / (c + 1.0f);
            }

            ctrFeatures.push_back(std::move(ctr));
        }

        // Replace this categorical feature with its CTR value(s) in the dataset
        // Mark the original feature as non-categorical (it becomes the first CTR's numeric values)
        // First, save the original hash map into all CTR features for this feature
        for (auto& ctr : ctrFeatures) {
            if (ctr.OrigFeatureIdx == f) {
                ctr.OrigCatHashMap = ds.CatHashMaps[f];
            }
        }
        ds.IsCategorical[f] = false;
        ds.HasNaN[f] = false;
        ds.CatHashMaps[f].clear();
        // Find the first CTR for this feature and use its values as replacement
        for (const auto& ctr : ctrFeatures) {
            if (ctr.OrigFeatureIdx == f) {
                ds.Features[f] = ctr.Values;
                ds.FeatureNames[f] = ctr.Name;
                break;
            }
        }
    }

    // Append additional CTR features (for multiclass: classes beyond the first)
    // Track which original features we've already placed the first CTR for
    std::unordered_set<ui32> firstCtrPlaced;
    for (const auto& ctr : ctrFeatures) {
        if (firstCtrPlaced.count(ctr.OrigFeatureIdx)) {
            // This is an additional CTR (e.g. class 1, 2, ... for multiclass)
            ds.Features.push_back(ctr.Values);
            ds.IsCategorical.push_back(false);
            ds.HasNaN.push_back(false);
            ds.FeatureNames.push_back(ctr.Name);
            ds.CatHashMaps.push_back({});
            ds.NumFeatures++;
        } else {
            firstCtrPlaced.insert(ctr.OrigFeatureIdx);
        }
    }

    return ctrFeatures;
}

// ============================================================================
// Model save: tree record and JSON serialization
// ============================================================================

struct TTreeRecord {
    std::vector<TObliviousSplitLevel> Splits;
    std::vector<TBestSplitProperties> SplitProps;
    std::vector<float> LeafValues;  // flat: numLeaves for dim=1, numLeaves*approxDim for multi
    ui32 Depth;
};

static std::string EscapeJsonString(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

void SaveModelJSON(
    const std::string& path,
    const std::vector<TTreeRecord>& allTrees,
    const TDataset& ds,
    const TQuantization& quant,
    const std::string& lossType,
    float lossParam,
    const TConfig& config,
    ui32 approxDim,
    ui32 numClasses,
    const std::vector<TCtrFeature>& ctrFeatures = {}
) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Error: Cannot open output file: %s\n", path.c_str());
        return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"format\": \"catboost-mlx-json\",\n");
    fprintf(f, "  \"version\": 1,\n");

    // model_info
    fprintf(f, "  \"model_info\": {\n");
    fprintf(f, "    \"loss_type\": \"%s\",\n", lossType.c_str());
    fprintf(f, "    \"loss_param\": %g,\n", lossParam);
    fprintf(f, "    \"learning_rate\": %g,\n", config.LearningRate);
    fprintf(f, "    \"l2_reg_lambda\": %g,\n", config.L2RegLambda);
    fprintf(f, "    \"approx_dimension\": %u,\n", approxDim);
    fprintf(f, "    \"num_classes\": %u,\n", numClasses);
    fprintf(f, "    \"num_trees\": %u,\n", static_cast<ui32>(allTrees.size()));
    fprintf(f, "    \"max_depth\": %u,\n", config.MaxDepth);
    fprintf(f, "    \"nan_mode\": \"%s\"\n", config.NanMode.c_str());
    fprintf(f, "  },\n");

    // features
    fprintf(f, "  \"features\": [\n");
    for (ui32 fi = 0; fi < ds.NumFeatures; ++fi) {
        fprintf(f, "    {\n");
        fprintf(f, "      \"index\": %u,\n", fi);
        std::string name = (fi < ds.FeatureNames.size() && !ds.FeatureNames[fi].empty())
            ? ds.FeatureNames[fi] : ("f" + std::to_string(fi));
        fprintf(f, "      \"name\": \"%s\",\n", EscapeJsonString(name).c_str());
        fprintf(f, "      \"is_categorical\": %s,\n", ds.IsCategorical[fi] ? "true" : "false");
        fprintf(f, "      \"has_nan\": %s,\n", ds.HasNaN[fi] ? "true" : "false");

        // borders
        fprintf(f, "      \"borders\": [");
        for (ui32 b = 0; b < quant.Borders[fi].size(); ++b) {
            if (b > 0) fprintf(f, ", ");
            fprintf(f, "%.8g", quant.Borders[fi][b]);
        }
        fprintf(f, "],\n");

        // cat_hash_map
        fprintf(f, "      \"cat_hash_map\": {");
        if (ds.IsCategorical[fi]) {
            bool first = true;
            for (const auto& [key, val] : ds.CatHashMaps[fi]) {
                if (!first) fprintf(f, ", ");
                fprintf(f, "\"%s\": %u", EscapeJsonString(key).c_str(), val);
                first = false;
            }
        }
        fprintf(f, "}\n");

        fprintf(f, "    }%s\n", (fi + 1 < ds.NumFeatures) ? "," : "");
    }
    fprintf(f, "  ],\n");

    // trees
    fprintf(f, "  \"trees\": [\n");
    for (ui32 ti = 0; ti < allTrees.size(); ++ti) {
        const auto& tree = allTrees[ti];
        fprintf(f, "    {\n");
        fprintf(f, "      \"depth\": %u,\n", tree.Depth);

        // splits
        fprintf(f, "      \"splits\": [\n");
        for (ui32 si = 0; si < tree.SplitProps.size(); ++si) {
            fprintf(f, "        {\"feature_idx\": %u, \"bin_threshold\": %u, \"is_one_hot\": %s}%s\n",
                    tree.SplitProps[si].FeatureId,
                    tree.SplitProps[si].BinId,
                    tree.Splits[si].IsOneHot ? "true" : "false",
                    (si + 1 < tree.SplitProps.size()) ? "," : "");
        }
        fprintf(f, "      ],\n");

        // leaf_values
        fprintf(f, "      \"leaf_values\": [");
        for (ui32 li = 0; li < tree.LeafValues.size(); ++li) {
            if (li > 0) fprintf(f, ", ");
            fprintf(f, "%.8g", tree.LeafValues[li]);
        }
        fprintf(f, "],\n");

        // split_gains
        fprintf(f, "      \"split_gains\": [");
        for (ui32 si = 0; si < tree.SplitProps.size(); ++si) {
            if (si > 0) fprintf(f, ", ");
            fprintf(f, "%.8g", tree.SplitProps[si].Gain);
        }
        fprintf(f, "]\n");

        fprintf(f, "    }%s\n", (ti + 1 < allTrees.size()) ? "," : "");
    }
    fprintf(f, "  ],\n");

    // feature_importance
    fprintf(f, "  \"feature_importance\": [\n");
    {
        std::vector<double> featureGain(ds.NumFeatures, 0.0);
        for (const auto& tree : allTrees) {
            for (ui32 level = 0; level < tree.SplitProps.size(); ++level) {
                ui32 featIdx = tree.SplitProps[level].FeatureId;
                if (featIdx < ds.NumFeatures) {
                    featureGain[featIdx] += tree.SplitProps[level].Gain;
                }
            }
        }
        double totalGain = 0.0;
        for (double g : featureGain) totalGain += g;

        std::vector<ui32> sortedIndices(ds.NumFeatures);
        std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                  [&](ui32 a, ui32 b) { return featureGain[a] > featureGain[b]; });

        bool first = true;
        for (ui32 rank = 0; rank < ds.NumFeatures; ++rank) {
            ui32 fi = sortedIndices[rank];
            if (featureGain[fi] <= 0.0) continue;
            if (!first) fprintf(f, ",\n");
            std::string name = (fi < ds.FeatureNames.size() && !ds.FeatureNames[fi].empty())
                ? ds.FeatureNames[fi] : ("f" + std::to_string(fi));
            double pct = (totalGain > 0) ? 100.0 * featureGain[fi] / totalGain : 0.0;
            fprintf(f, "    {\"index\": %u, \"name\": \"%s\", \"gain\": %.6f, \"percent\": %.2f}",
                    fi, EscapeJsonString(name).c_str(), featureGain[fi], pct);
            first = false;
        }
        if (!first) fprintf(f, "\n");
    }
    fprintf(f, "  ]%s\n", ctrFeatures.empty() ? "" : ",");

    // ctr_features (only if CTR was used)
    if (!ctrFeatures.empty()) {
        fprintf(f, "  \"ctr_features\": [\n");
        for (size_t ci = 0; ci < ctrFeatures.size(); ++ci) {
            const auto& ctr = ctrFeatures[ci];
            fprintf(f, "    {\n");
            fprintf(f, "      \"original_feature_idx\": %u,\n", ctr.OrigFeatureIdx);
            fprintf(f, "      \"name\": \"%s\",\n", EscapeJsonString(ctr.Name).c_str());
            fprintf(f, "      \"prior\": %g,\n", ctr.Prior);
            fprintf(f, "      \"class_idx\": %u,\n", ctr.ClassIdx);
            fprintf(f, "      \"default_value\": %.8g,\n", ctr.DefaultCtr);
            fprintf(f, "      \"final_values\": {");
            bool first = true;
            for (const auto& [bin, val] : ctr.FinalCtrValues) {
                if (!first) fprintf(f, ", ");
                fprintf(f, "\"%u\": %.8g", bin, val);
                first = false;
            }
            fprintf(f, "},\n");
            fprintf(f, "      \"cat_hash_map\": {");
            first = true;
            for (const auto& [catStr, catBin] : ctr.OrigCatHashMap) {
                if (!first) fprintf(f, ", ");
                fprintf(f, "\"%s\": %u", EscapeJsonString(catStr).c_str(), catBin);
                first = false;
            }
            fprintf(f, "}\n");
            fprintf(f, "    }%s\n", (ci + 1 < ctrFeatures.size()) ? "," : "");
        }
        fprintf(f, "  ]\n");
    }

    fprintf(f, "}\n");
    fclose(f);
}

// ============================================================================
// Training function (extracted for reuse in cross-validation)
// ============================================================================

struct TTrainResult {
    std::vector<TTreeRecord> Trees;
    float FinalTrainLoss = 0.0f;
    float FinalTestLoss = 0.0f;
    ui32 BestIteration = 0;
    ui32 TreesBuilt = 0;
};

// ============================================================================
// Snapshot save/restore for training resume
// ============================================================================

struct TSnapshot {
    ui32 Iteration = 0;
    std::vector<TTreeRecord> Trees;
    std::vector<float> TrainCursor;  // flat [trainDocs] or [K*trainDocs]
    std::vector<float> ValCursor;    // flat (empty if no val)
    float BestValLoss = 1e30f;
    ui32 BestIteration = 0;
    ui32 NoImprovementCount = 0;
    std::string RngState;
    bool Valid = false;
};

void SaveSnapshot(
    const std::string& path,
    ui32 currentIteration,
    const std::vector<TTreeRecord>& trees,
    const mx::array& cursor, ui32 trainDocs,
    const mx::array& valCursor, ui32 valDocs,
    ui32 approxDim,
    float bestValLoss, ui32 bestIteration, ui32 noImprovementCount,
    const std::mt19937& rng
) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Warning: cannot save snapshot to %s\n", path.c_str()); return; }

    fprintf(f, "{\n");
    fprintf(f, "  \"snapshot_version\": 1,\n");
    fprintf(f, "  \"iteration\": %u,\n", currentIteration);
    fprintf(f, "  \"num_trees\": %u,\n", static_cast<ui32>(trees.size()));
    fprintf(f, "  \"approx_dim\": %u,\n", approxDim);
    fprintf(f, "  \"best_val_loss\": %.10g,\n", bestValLoss);
    fprintf(f, "  \"best_iteration\": %u,\n", bestIteration);
    fprintf(f, "  \"no_improvement_count\": %u,\n", noImprovementCount);

    // RNG state
    std::ostringstream rngOss;
    rngOss << rng;
    fprintf(f, "  \"rng_state\": \"%s\",\n", rngOss.str().c_str());

    // Train cursor
    mx::eval(cursor);
    ui32 cursorSize = (approxDim == 1) ? trainDocs : approxDim * trainDocs;
    const float* cPtr = cursor.data<float>();
    fprintf(f, "  \"train_cursor\": [");
    for (ui32 i = 0; i < cursorSize; ++i) {
        if (i > 0) fprintf(f, ",");
        fprintf(f, "%.10g", cPtr[i]);
    }
    fprintf(f, "],\n");

    // Val cursor
    fprintf(f, "  \"val_cursor\": [");
    if (valDocs > 0) {
        mx::eval(valCursor);
        ui32 valCursorSize = (approxDim == 1) ? valDocs : approxDim * valDocs;
        const float* vPtr = valCursor.data<float>();
        for (ui32 i = 0; i < valCursorSize; ++i) {
            if (i > 0) fprintf(f, ",");
            fprintf(f, "%.10g", vPtr[i]);
        }
    }
    fprintf(f, "],\n");

    // Trees
    fprintf(f, "  \"trees\": [\n");
    for (ui32 t = 0; t < trees.size(); ++t) {
        const auto& tree = trees[t];
        fprintf(f, "    {\"depth\": %u, \"splits\": [", tree.Depth);
        for (ui32 s = 0; s < tree.Splits.size(); ++s) {
            const auto& sp = tree.Splits[s];
            if (s > 0) fprintf(f, ",");
            fprintf(f, "{\"col\":%llu,\"shift\":%u,\"mask\":%u,\"bin\":%u,\"onehot\":%s}",
                    sp.FeatureColumnIdx, sp.Shift, sp.Mask, sp.BinThreshold,
                    sp.IsOneHot ? "true" : "false");
        }
        fprintf(f, "], \"split_props\": [");
        for (ui32 s = 0; s < tree.SplitProps.size(); ++s) {
            if (s > 0) fprintf(f, ",");
            fprintf(f, "{\"feat\":%u,\"bin\":%u,\"gain\":%.10g}",
                    tree.SplitProps[s].FeatureId, tree.SplitProps[s].BinId, tree.SplitProps[s].Gain);
        }
        fprintf(f, "], \"leaf_values\": [");
        for (ui32 l = 0; l < tree.LeafValues.size(); ++l) {
            if (l > 0) fprintf(f, ",");
            fprintf(f, "%.10g", tree.LeafValues[l]);
        }
        fprintf(f, "]}%s\n", (t + 1 < trees.size()) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

TSnapshot LoadSnapshot(const std::string& path) {
    TSnapshot snap;
    std::ifstream file(path);
    if (!file.is_open()) return snap;

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();

    // Simple JSON parsing: extract fields by key
    auto extractUint = [&](const std::string& key) -> ui32 {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = content.find(':', pos);
        return static_cast<ui32>(std::atoi(content.c_str() + pos + 1));
    };
    auto extractFloat = [&](const std::string& key) -> float {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0.0f;
        pos = content.find(':', pos);
        return std::atof(content.c_str() + pos + 1);
    };
    auto extractString = [&](const std::string& key) -> std::string {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = content.find('"', pos + key.size() + 2);
        if (pos == std::string::npos) return "";
        auto end = content.find('"', pos + 1);
        return content.substr(pos + 1, end - pos - 1);
    };
    auto extractFloatArray = [&](const std::string& key) -> std::vector<float> {
        std::vector<float> result;
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return result;
        auto start = content.find('[', pos);
        auto end = content.find(']', start);
        if (start == std::string::npos || end == std::string::npos) return result;
        std::string arr = content.substr(start + 1, end - start - 1);
        std::stringstream ss(arr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                result.push_back(std::stof(token));
            }
        }
        return result;
    };

    ui32 version = extractUint("snapshot_version");
    if (version != 1) return snap;

    snap.Iteration = extractUint("iteration");
    ui32 numTrees = extractUint("num_trees");
    ui32 approxDim = extractUint("approx_dim");
    snap.BestValLoss = extractFloat("best_val_loss");
    snap.BestIteration = extractUint("best_iteration");
    snap.NoImprovementCount = extractUint("no_improvement_count");
    snap.RngState = extractString("rng_state");
    snap.TrainCursor = extractFloatArray("train_cursor");
    snap.ValCursor = extractFloatArray("val_cursor");

    // Parse trees: find "trees" array, then parse each tree object
    auto treesPos = content.find("\"trees\"");
    if (treesPos != std::string::npos) {
        auto arrStart = content.find('[', treesPos);
        if (arrStart != std::string::npos) {
            // Find matching closing bracket (accounting for nested arrays)
            int depth = 0;
            size_t arrEnd = arrStart;
            for (size_t i = arrStart; i < content.size(); ++i) {
                if (content[i] == '[') depth++;
                else if (content[i] == ']') { depth--; if (depth == 0) { arrEnd = i; break; } }
            }

            // Parse each tree: find { ... } blocks
            std::string treesStr = content.substr(arrStart + 1, arrEnd - arrStart - 1);
            size_t searchPos = 0;
            while (snap.Trees.size() < numTrees) {
                auto treeStart = treesStr.find('{', searchPos);
                if (treeStart == std::string::npos) break;

                // Find matching closing brace
                int bd = 0;
                size_t treeEnd = treeStart;
                for (size_t i = treeStart; i < treesStr.size(); ++i) {
                    if (treesStr[i] == '{') bd++;
                    else if (treesStr[i] == '}') { bd--; if (bd == 0) { treeEnd = i; break; } }
                }

                std::string treeStr = treesStr.substr(treeStart, treeEnd - treeStart + 1);
                TTreeRecord record;

                // Parse depth
                auto dPos = treeStr.find("\"depth\"");
                if (dPos != std::string::npos) {
                    auto cp = treeStr.find(':', dPos);
                    record.Depth = std::atoi(treeStr.c_str() + cp + 1);
                }

                // Parse leaf_values
                auto lvPos = treeStr.find("\"leaf_values\"");
                if (lvPos != std::string::npos) {
                    auto lStart = treeStr.find('[', lvPos);
                    auto lEnd = treeStr.find(']', lStart);
                    std::string lvStr = treeStr.substr(lStart + 1, lEnd - lStart - 1);
                    std::stringstream lvSs(lvStr);
                    std::string tok;
                    while (std::getline(lvSs, tok, ',')) {
                        if (!tok.empty()) record.LeafValues.push_back(std::stof(tok));
                    }
                }

                // Parse splits array
                auto splitsPos = treeStr.find("\"splits\"");
                if (splitsPos != std::string::npos) {
                    auto sStart = treeStr.find('[', splitsPos);
                    auto sEnd = treeStr.find(']', sStart);
                    std::string splitsStr = treeStr.substr(sStart + 1, sEnd - sStart - 1);
                    // Parse each split object
                    size_t sp = 0;
                    while (true) {
                        auto sb = splitsStr.find('{', sp);
                        if (sb == std::string::npos) break;
                        auto se = splitsStr.find('}', sb);
                        std::string splitObj = splitsStr.substr(sb, se - sb + 1);

                        TObliviousSplitLevel split;
                        auto extractSplitField = [&](const std::string& k) -> long long {
                            auto p2 = splitObj.find("\"" + k + "\"");
                            if (p2 == std::string::npos) return 0;
                            auto cp2 = splitObj.find(':', p2);
                            return std::atoll(splitObj.c_str() + cp2 + 1);
                        };
                        split.FeatureColumnIdx = static_cast<ui32>(extractSplitField("col"));
                        split.Shift = static_cast<ui32>(extractSplitField("shift"));
                        split.Mask = static_cast<ui32>(extractSplitField("mask"));
                        split.BinThreshold = static_cast<ui32>(extractSplitField("bin"));
                        split.IsOneHot = (splitObj.find("\"onehot\":true") != std::string::npos);
                        record.Splits.push_back(split);
                        sp = se + 1;
                    }
                }

                // Parse split_props array
                auto propsPos = treeStr.find("\"split_props\"");
                if (propsPos != std::string::npos) {
                    auto pStart = treeStr.find('[', propsPos);
                    auto pEnd = treeStr.find(']', pStart);
                    std::string propsStr = treeStr.substr(pStart + 1, pEnd - pStart - 1);
                    size_t pp = 0;
                    while (true) {
                        auto pb = propsStr.find('{', pp);
                        if (pb == std::string::npos) break;
                        auto pe = propsStr.find('}', pb);
                        std::string propObj = propsStr.substr(pb, pe - pb + 1);

                        TBestSplitProperties prop;
                        auto extractPropField = [&](const std::string& k) -> float {
                            auto p2 = propObj.find("\"" + k + "\"");
                            if (p2 == std::string::npos) return 0.0f;
                            auto cp2 = propObj.find(':', p2);
                            return std::atof(propObj.c_str() + cp2 + 1);
                        };
                        prop.FeatureId = static_cast<ui32>(extractPropField("feat"));
                        prop.BinId = static_cast<ui32>(extractPropField("bin"));
                        prop.Gain = extractPropField("gain");
                        record.SplitProps.push_back(prop);
                        pp = pe + 1;
                    }
                }

                snap.Trees.push_back(std::move(record));
                searchPos = treeEnd + 1;
            }
        }
    }

    snap.Valid = (snap.Trees.size() == numTrees && !snap.TrainCursor.empty());
    return snap;
}

TTrainResult RunTraining(
    const TConfig& config,
    const mx::array& compressedData, ui32 trainDocs,
    const mx::array& targetsArr,
    const mx::array& valCompressedData, ui32 valDocs,
    const mx::array& valTargetsArr,
    const TPackedData& packed,
    const std::string& lossType, float lossParam,
    ui32 approxDim, ui32 numClasses,
    bool printProgress,
    const std::vector<float>& trainTargetsVec = {},        // needed for ranking grad scatter
    const std::vector<ui32>& trainGroupOffsets = {},        // group offsets for ranking
    ui32 trainNumGroups = 0,
    const std::vector<float>& valTargetsVec = {},           // for NDCG on validation
    const std::vector<ui32>& valGroupOffsets = {},
    ui32 valNumGroups = 0,
    const std::vector<float>& sampleWeights = {},            // per-sample weights (empty = uniform)
    const std::string& snapshotPath = "",                    // snapshot file for save/resume
    ui32 snapshotInterval = 1                                // save every N iterations
) {
    TTrainResult result;

    // Initialize cursor
    mx::array cursor = (approxDim == 1)
        ? mx::zeros({static_cast<int>(trainDocs)}, mx::float32)
        : mx::zeros({static_cast<int>(approxDim), static_cast<int>(trainDocs)}, mx::float32);
    mx::array valCursor = mx::array(0.0f);
    if (valDocs > 0) {
        valCursor = (approxDim == 1)
            ? mx::zeros({static_cast<int>(valDocs)}, mx::float32)
            : mx::zeros({static_cast<int>(approxDim), static_cast<int>(valDocs)}, mx::float32);
    }

    // Random number generator for subsampling
    std::mt19937 rng(config.RandomSeed);

    result.Trees.reserve(config.NumIterations);

    // Early stopping state
    float bestValLoss = std::numeric_limits<float>::infinity();
    ui32 bestIteration = 0;
    ui32 noImprovementCount = 0;

    // Snapshot resume
    ui32 startIteration = 0;
    if (!snapshotPath.empty()) {
        auto snap = LoadSnapshot(snapshotPath);
        if (snap.Valid) {
            result.Trees = std::move(snap.Trees);
            result.TreesBuilt = result.Trees.size();
            startIteration = snap.Iteration + 1;
            bestValLoss = snap.BestValLoss;
            bestIteration = snap.BestIteration;
            noImprovementCount = snap.NoImprovementCount;

            // Restore cursor
            if (!snap.TrainCursor.empty()) {
                if (approxDim == 1) {
                    cursor = mx::array(snap.TrainCursor.data(), {static_cast<int>(trainDocs)}, mx::float32);
                } else {
                    cursor = mx::array(snap.TrainCursor.data(),
                        {static_cast<int>(approxDim), static_cast<int>(trainDocs)}, mx::float32);
                }
                mx::eval(cursor);
            }
            if (!snap.ValCursor.empty() && valDocs > 0) {
                if (approxDim == 1) {
                    valCursor = mx::array(snap.ValCursor.data(), {static_cast<int>(valDocs)}, mx::float32);
                } else {
                    valCursor = mx::array(snap.ValCursor.data(),
                        {static_cast<int>(approxDim), static_cast<int>(valDocs)}, mx::float32);
                }
                mx::eval(valCursor);
            }

            // Restore RNG state
            if (!snap.RngState.empty()) {
                std::istringstream iss(snap.RngState);
                iss >> rng;
            }

            if (printProgress) printf("Resumed from snapshot: iteration %u (%u trees)\n",
                                       snap.Iteration, static_cast<ui32>(result.Trees.size()));
        }
    }

    // Ranking: precompute pairs for PairLogit (static), or generate per-iter for YetiRank
    bool isRanking = (lossType == "pairlogit" || lossType == "yetirank");
    std::vector<TPair> pairLogitPairs;
    if (lossType == "pairlogit" && !trainGroupOffsets.empty()) {
        pairLogitPairs = GeneratePairLogitPairs(trainTargetsVec, trainGroupOffsets, trainNumGroups);
        if (printProgress) printf("PairLogit: %zu pairs from %u groups\n", pairLogitPairs.size(), trainNumGroups);
    }
    // For val set NDCG, precompute PairLogit pairs too
    std::vector<TPair> valPairLogitPairs;
    if (lossType == "pairlogit" && !valGroupOffsets.empty()) {
        valPairLogitPairs = GeneratePairLogitPairs(valTargetsVec, valGroupOffsets, valNumGroups);
    }

    for (ui32 iter = startIteration; iter < config.NumIterations; ++iter) {
        auto iterStart = std::chrono::steady_clock::now();

        // --- Bootstrap weights ---
        // Determine effective bootstrap type
        std::string bootstrapType = config.BootstrapType;
        if (bootstrapType == "no" && config.SubsampleRatio < 1.0f) {
            bootstrapType = "bernoulli";  // --subsample without explicit type defaults to Bernoulli
        }

        bool useBootstrap = (bootstrapType != "no");

        // --- Feature subsampling ---
        std::vector<bool> featureMask;
        bool useColsample = (config.ColsampleByTree < 1.0f);
        if (useColsample) {
            ui32 numFeats = packed.Features.size();
            ui32 selectCount = static_cast<ui32>(std::ceil(config.ColsampleByTree * numFeats));
            selectCount = std::min(selectCount, numFeats);
            std::vector<ui32> featIndices(numFeats);
            std::iota(featIndices.begin(), featIndices.end(), 0);
            std::shuffle(featIndices.begin(), featIndices.end(), rng);
            featureMask.resize(numFeats, false);
            for (ui32 i = 0; i < selectCount; ++i) featureMask[featIndices[i]] = true;
        }

        // Step 1: Compute gradients per dimension
        std::vector<mx::array> dimGrads, dimHess;
        dimGrads.reserve(approxDim);
        dimHess.reserve(approxDim);
        for (ui32 k = 0; k < approxDim; ++k) {
            dimGrads.push_back(mx::zeros({static_cast<int>(trainDocs)}, mx::float32));
            dimHess.push_back(mx::zeros({static_cast<int>(trainDocs)}, mx::float32));
        }

        if (lossType == "rmse") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            dimGrads[0] = mx::subtract(flatCursor, targetsArr);
            dimHess[0] = mx::ones({static_cast<int>(trainDocs)}, mx::float32);
        } else if (lossType == "mae") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            dimGrads[0] = mx::sign(mx::subtract(flatCursor, targetsArr));
            dimHess[0] = mx::ones({static_cast<int>(trainDocs)}, mx::float32);
        } else if (lossType == "quantile") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            auto diff = mx::subtract(flatCursor, targetsArr);
            auto isPositive = mx::greater(diff, mx::array(0.0f));
            dimGrads[0] = mx::where(isPositive, mx::array(1.0f - lossParam), mx::array(-lossParam));
            dimHess[0] = mx::ones({static_cast<int>(trainDocs)}, mx::float32);
        } else if (lossType == "huber") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            auto diff = mx::subtract(flatCursor, targetsArr);
            auto absDiff = mx::abs(diff);
            auto isSmall = mx::less_equal(absDiff, mx::array(lossParam));
            dimGrads[0] = mx::where(isSmall, diff, mx::multiply(mx::array(lossParam), mx::sign(diff)));
            dimHess[0] = mx::where(isSmall,
                mx::ones({static_cast<int>(trainDocs)}, mx::float32),
                mx::full({static_cast<int>(trainDocs)}, 1e-6f, mx::float32));
        } else if (lossType == "logloss") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            auto sigmoid = mx::sigmoid(flatCursor);
            dimGrads[0] = mx::subtract(sigmoid, targetsArr);
            dimHess[0] = mx::maximum(
                mx::multiply(sigmoid, mx::subtract(mx::array(1.0f), sigmoid)),
                mx::array(1e-16f)
            );
        } else if (lossType == "poisson") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            auto expPred = mx::exp(flatCursor);
            dimGrads[0] = mx::subtract(expPred, targetsArr);
            dimHess[0] = mx::maximum(expPred, mx::array(1e-6f));
        } else if (lossType == "tweedie") {
            float p = lossParam;
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            dimGrads[0] = mx::add(
                mx::multiply(mx::negative(targetsArr), mx::exp(mx::multiply(flatCursor, mx::array(1.0f - p)))),
                mx::exp(mx::multiply(flatCursor, mx::array(2.0f - p))));
            auto hess = mx::add(
                mx::multiply(mx::multiply(mx::negative(targetsArr), mx::array(1.0f - p)),
                             mx::exp(mx::multiply(flatCursor, mx::array(1.0f - p)))),
                mx::multiply(mx::array(2.0f - p),
                             mx::exp(mx::multiply(flatCursor, mx::array(2.0f - p)))));
            dimHess[0] = mx::maximum(hess, mx::array(1e-6f));
        } else if (lossType == "mape") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            auto absTarget = mx::maximum(mx::abs(targetsArr), mx::array(1e-6f));
            dimGrads[0] = mx::divide(mx::sign(mx::subtract(flatCursor, targetsArr)), absTarget);
            dimHess[0] = mx::divide(mx::ones({static_cast<int>(trainDocs)}, mx::float32), absTarget);
        } else if (lossType == "pairlogit" || lossType == "yetirank") {
            // Ranking: scatter pairwise gradients to per-doc arrays
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            mx::eval(flatCursor);
            const float* predsPtr = flatCursor.data<float>();

            std::vector<TPair>* activePairs = nullptr;
            std::vector<TPair> yetiPairs;  // temporary storage for YetiRank pairs
            if (lossType == "pairlogit") {
                activePairs = &pairLogitPairs;
            } else {
                // YetiRank: generate fresh pairs each iteration
                yetiPairs = GenerateYetiRankPairs(trainTargetsVec, trainGroupOffsets, trainNumGroups, rng);
                activePairs = &yetiPairs;
            }

            std::vector<float> cpuGrads, cpuHess;
            ScatterPairwiseGradients(*activePairs, predsPtr, trainDocs, cpuGrads, cpuHess);
            dimGrads[0] = mx::array(cpuGrads.data(), {static_cast<int>(trainDocs)}, mx::float32);
            dimHess[0] = mx::array(cpuHess.data(), {static_cast<int>(trainDocs)}, mx::float32);
        } else {  // multiclass
            auto sm = ComputeSoftmax(cursor, approxDim, trainDocs);
            auto targetInt = mx::astype(targetsArr, mx::uint32);
            mx::eval(sm.Probs);

            for (ui32 k = 0; k < approxDim; ++k) {
                auto isClass = mx::astype(
                    mx::equal(targetInt, mx::array(static_cast<uint32_t>(k))),
                    mx::float32
                );
                auto probK = mx::reshape(
                    mx::slice(sm.Probs, {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(trainDocs)}),
                    {static_cast<int>(trainDocs)}
                );
                dimGrads[k] = mx::subtract(probK, isClass);
                dimHess[k] = mx::maximum(
                    mx::multiply(probK, mx::subtract(mx::array(1.0f), probK)),
                    mx::array(1e-16f)
                );
            }
        }

        for (ui32 k = 0; k < approxDim; ++k) mx::eval({dimGrads[k], dimHess[k]});

        // --- Apply sample weights to gradients/hessians ---
        if (!sampleWeights.empty()) {
            auto sampleWeightsArr = mx::array(sampleWeights.data(), {static_cast<int>(trainDocs)}, mx::float32);
            for (ui32 k = 0; k < approxDim; ++k) {
                dimGrads[k] = mx::multiply(dimGrads[k], sampleWeightsArr);
                dimHess[k] = mx::multiply(dimHess[k], sampleWeightsArr);
            }
            for (ui32 k = 0; k < approxDim; ++k) mx::eval({dimGrads[k], dimHess[k]});
        }

        // --- Apply bootstrap weights to gradients/hessians ---
        if (useBootstrap) {
            std::vector<float> weights(trainDocs, 1.0f);

            if (bootstrapType == "bayesian") {
                float invTemp = 1.0f / std::max(config.BaggingTemperature, 1e-6f);
                std::exponential_distribution<float> expDist(invTemp);
                for (ui32 d = 0; d < trainDocs; ++d) {
                    weights[d] = expDist(rng);
                }
            } else if (bootstrapType == "bernoulli") {
                float p = config.SubsampleRatio;
                std::bernoulli_distribution bernDist(p);
                for (ui32 d = 0; d < trainDocs; ++d) {
                    weights[d] = bernDist(rng) ? 1.0f : 0.0f;
                }
            } else if (bootstrapType == "mvs") {
                // Minimum Variance Sampling: include top docs by gradient magnitude,
                // weight remaining proportionally to gradient magnitude
                float mvsReg = config.MvsReg;
                std::vector<float> gradMag(trainDocs);
                mx::eval(dimGrads[0]);
                const float* g0 = dimGrads[0].data<float>();
                for (ui32 d = 0; d < trainDocs; ++d) {
                    float mag = 0.0f;
                    if (approxDim == 1) {
                        mag = std::fabs(g0[d]);
                    } else {
                        for (ui32 k = 0; k < approxDim; ++k) {
                            mx::eval(dimGrads[k]);
                            const float* gk = dimGrads[k].data<float>();
                            mag += gk[d] * gk[d];
                        }
                        mag = std::sqrt(mag);
                    }
                    gradMag[d] = mag + mvsReg;
                }

                // Find threshold: include top fraction by gradient magnitude
                std::vector<float> sortedMag = gradMag;
                std::sort(sortedMag.begin(), sortedMag.end(), std::greater<float>());
                ui32 topCount = static_cast<ui32>(std::ceil(config.SubsampleRatio * trainDocs));
                topCount = std::min(topCount, trainDocs);
                float threshold = (topCount < trainDocs) ? sortedMag[topCount - 1] : 0.0f;

                for (ui32 d = 0; d < trainDocs; ++d) {
                    if (gradMag[d] >= threshold) {
                        weights[d] = 1.0f;
                    } else if (threshold > 0.0f) {
                        weights[d] = gradMag[d] / threshold;
                    } else {
                        weights[d] = 1.0f;
                    }
                }
            }

            auto weightsArr = mx::array(weights.data(), {static_cast<int>(trainDocs)}, mx::float32);
            for (ui32 k = 0; k < approxDim; ++k) {
                dimGrads[k] = mx::multiply(dimGrads[k], weightsArr);
                dimHess[k] = mx::multiply(dimHess[k], weightsArr);
            }
            for (ui32 k = 0; k < approxDim; ++k) mx::eval({dimGrads[k], dimHess[k]});
        }

        // Step 2: Greedy tree structure search
        auto partitions = mx::zeros({static_cast<int>(trainDocs)}, mx::uint32);
        std::vector<TObliviousSplitLevel> splits;
        std::vector<TBestSplitProperties> splitProps;

        for (ui32 depth = 0; depth < config.MaxDepth; ++depth) {
            ui32 numPartitions = 1u << depth;
            auto layout = ComputePartitionLayout(partitions, trainDocs, numPartitions);

            // Compute per-dim histograms and partition stats
            std::vector<std::vector<float>> perDimHistData(approxDim);
            std::vector<std::vector<TPartitionStatistics>> perDimPartStats(approxDim);

            mx::eval(partitions);
            const uint32_t* partsPtr = partitions.data<uint32_t>();

            for (ui32 k = 0; k < approxDim; ++k) {
                auto statsK = mx::concatenate({
                    mx::reshape(dimGrads[k], {1, static_cast<int>(trainDocs)}),
                    mx::reshape(dimHess[k], {1, static_cast<int>(trainDocs)})
                }, 0);
                statsK = mx::reshape(statsK, {static_cast<int>(2 * trainDocs)});

                auto hist = DispatchHistogram(
                    compressedData, statsK,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    packed.Features, packed.NumUi32PerDoc,
                    packed.TotalBinFeatures, numPartitions, trainDocs
                );
                mx::eval(hist);
                const float* hData = hist.data<float>();
                perDimHistData[k].assign(hData, hData + numPartitions * 2 * packed.TotalBinFeatures);

                mx::eval(dimGrads[k]);
                mx::eval(dimHess[k]);
                const float* gPtr = dimGrads[k].data<float>();
                const float* hPtr = dimHess[k].data<float>();

                perDimPartStats[k].resize(numPartitions);
                for (ui32 d = 0; d < trainDocs; ++d) {
                    ui32 p = partsPtr[d];
                    if (p < numPartitions) {
                        perDimPartStats[k][p].Sum += gPtr[d];
                        perDimPartStats[k][p].Weight += hPtr[d];
                    }
                }
            }

            // Build count histogram for min-data-in-leaf
            std::vector<std::vector<ui32>> countHist;
            std::vector<ui32> partDocCounts;
            if (config.MinDataInLeaf > 1) {
                countHist.assign(numPartitions, std::vector<ui32>(packed.TotalBinFeatures, 0));
                partDocCounts.assign(numPartitions, 0);

                mx::eval(compressedData);
                const uint32_t* cdPtr = compressedData.data<uint32_t>();

                for (ui32 d = 0; d < trainDocs; ++d) {
                    ui32 p = partsPtr[d];
                    if (p >= numPartitions) continue;
                    partDocCounts[p]++;
                    const uint32_t* docData = cdPtr + d * packed.NumUi32PerDoc;
                    for (ui32 fi = 0; fi < packed.Features.size(); ++fi) {
                        const auto& f = packed.Features[fi];
                        uint32_t word = docData[f.Offset];
                        uint32_t binVal = (word >> f.Shift) & (f.Mask >> f.Shift);
                        // binVal is 0-based: bin 0 = "below first border" (or NaN bin)
                        // The histogram bins correspond to binVal 1..Folds (the kernel uses +1 offset)
                        if (binVal > 0 && binVal <= f.Folds) {
                            countHist[p][f.FirstFoldIndex + binVal - 1]++;
                        }
                    }
                }
            }

            auto bestSplit = FindBestSplit(
                perDimHistData, perDimPartStats,
                packed.Features, packed.TotalBinFeatures,
                config.L2RegLambda, numPartitions, featureMask,
                config.MinDataInLeaf, countHist, partDocCounts,
                config.MonotoneConstraints
            );

            if (!bestSplit.Defined()) break;

            // Record split
            const auto& feat = packed.Features[bestSplit.FeatureId];
            TObliviousSplitLevel split;
            split.FeatureColumnIdx = static_cast<ui32>(feat.Offset);
            split.Shift = feat.Shift;
            split.Mask = feat.Mask >> feat.Shift;
            split.BinThreshold = bestSplit.BinId;
            split.IsOneHot = feat.OneHotFeature;
            splits.push_back(split);
            splitProps.push_back(bestSplit);

            // Update partitions
            auto column = mx::slice(compressedData,
                {0, static_cast<int>(split.FeatureColumnIdx)},
                {static_cast<int>(trainDocs), static_cast<int>(split.FeatureColumnIdx + 1)});
            column = mx::reshape(column, {static_cast<int>(trainDocs)});
            auto featureValues = mx::bitwise_and(
                mx::right_shift(column, mx::array(static_cast<uint32_t>(split.Shift), mx::uint32)),
                mx::array(static_cast<uint32_t>(split.Mask), mx::uint32));
            auto goRight = feat.OneHotFeature
                ? mx::equal(featureValues, mx::array(static_cast<uint32_t>(split.BinThreshold), mx::uint32))
                : mx::greater(featureValues, mx::array(static_cast<uint32_t>(split.BinThreshold), mx::uint32));
            auto bits = mx::left_shift(mx::astype(goRight, mx::uint32), mx::array(static_cast<uint32_t>(depth), mx::uint32));
            partitions = mx::bitwise_or(partitions, bits);
            mx::eval(partitions);
        }

        if (splits.empty()) {
            if (printProgress) printf("iter=%u: no valid split, stopping\n", iter);
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
            for (ui32 d = 0; d < trainDocs; ++d) {
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
                for (ui32 d = 0; d < trainDocs; ++d) {
                    ui32 leaf = leafAssign[d];
                    if (leaf < numLeaves) { gSums[leaf] += gp[d]; hSums[leaf] += hp[d]; }
                }
                for (ui32 i = 0; i < numLeaves; ++i)
                    interleaved[i * approxDim + k] = -config.LearningRate * gSums[i] / (hSums[i] + config.L2RegLambda);
            }
            leafValues = mx::array(interleaved.data(),
                {static_cast<int>(numLeaves), static_cast<int>(approxDim)}, mx::float32);
        }

        // Post-tree monotone constraint adjustment
        if (!config.MonotoneConstraints.empty() && approxDim == 1) {
            mx::eval(leafValues);
            std::vector<float> lv(leafValues.data<float>(), leafValues.data<float>() + numLeaves);
            for (ui32 level = 0; level < splits.size(); ++level) {
                ui32 featId = splitProps[level].FeatureId;
                if (featId >= config.MonotoneConstraints.size()) continue;
                int constraint = config.MonotoneConstraints[featId];
                if (constraint == 0) continue;

                for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
                    if (leaf & (1u << level)) continue;  // only process left child
                    ui32 leftLeaf = leaf;
                    ui32 rightLeaf = leaf | (1u << level);
                    float vl = lv[leftLeaf], vr = lv[rightLeaf];
                    if (constraint == 1 && vr < vl) {
                        float avg = 0.5f * (vl + vr);
                        lv[leftLeaf] = avg; lv[rightLeaf] = avg;
                    } else if (constraint == -1 && vl < vr) {
                        float avg = 0.5f * (vl + vr);
                        lv[leftLeaf] = avg; lv[rightLeaf] = avg;
                    }
                }
            }
            leafValues = mx::array(lv.data(), {static_cast<int>(numLeaves)}, mx::float32);
        }

        // Record tree
        {
            TTreeRecord record;
            record.Splits = splits;
            record.SplitProps = splitProps;
            record.Depth = static_cast<ui32>(splits.size());
            mx::eval(leafValues);
            const float* lvPtr = leafValues.data<float>();
            ui32 totalFloats = (approxDim == 1) ? numLeaves : numLeaves * approxDim;
            record.LeafValues.assign(lvPtr, lvPtr + totalFloats);
            result.Trees.push_back(std::move(record));
        }

        // Step 4: Apply tree to training data
        auto docLeafValues = mx::take(leafValues, mx::astype(partitions, mx::int32), 0);
        if (approxDim > 1) {
            docLeafValues = mx::transpose(docLeafValues);  // [K, numDocs]
            cursor = mx::add(cursor, docLeafValues);
        } else {
            cursor = mx::add(mx::reshape(cursor, {static_cast<int>(trainDocs)}), docLeafValues);
        }
        mx::eval(cursor);
        result.TreesBuilt++;

        // Apply tree to validation data
        if (valDocs > 0) {
            auto valLeafIndices = ComputeLeafIndices(valCompressedData, splits, valDocs);
            auto valDocLeafValues = mx::take(leafValues, mx::astype(valLeafIndices, mx::int32), 0);
            if (approxDim > 1) {
                valDocLeafValues = mx::transpose(valDocLeafValues);
                valCursor = mx::add(valCursor, valDocLeafValues);
            } else {
                valCursor = mx::add(mx::reshape(valCursor, {static_cast<int>(valDocs)}), valDocLeafValues);
            }
            mx::eval(valCursor);
        }

        // Step 5: Report loss
        auto iterEnd = std::chrono::steady_clock::now();
        auto iterMs = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart).count();

        if (printProgress && (config.Verbose || iter % 10 == 0 || iter == config.NumIterations - 1)) {
            float trainLoss;
            float trainNDCG = 0.0f;
            if (isRanking) {
                auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
                mx::eval(flatCursor);
                const float* predsPtr = flatCursor.data<float>();
                const auto& pairs = (lossType == "pairlogit") ? pairLogitPairs
                    : GenerateYetiRankPairs(trainTargetsVec, trainGroupOffsets, trainNumGroups, rng);
                trainLoss = ComputePairLogitLoss(pairs, predsPtr, trainDocs);
                trainNDCG = ComputeNDCG(trainTargetsVec, predsPtr, trainGroupOffsets, trainNumGroups);
            } else {
                trainLoss = ComputeLossValue(cursor, targetsArr, lossType, lossParam, approxDim, trainDocs);
            }
            result.FinalTrainLoss = trainLoss;

            if (valDocs > 0) {
                float valLoss;
                float valNDCG = 0.0f;
                if (isRanking) {
                    auto flatVal = mx::reshape(valCursor, {static_cast<int>(valDocs)});
                    mx::eval(flatVal);
                    const float* valPredsPtr = flatVal.data<float>();
                    const auto& vPairs = (lossType == "pairlogit") ? valPairLogitPairs
                        : GenerateYetiRankPairs(valTargetsVec, valGroupOffsets, valNumGroups, rng);
                    valLoss = ComputePairLogitLoss(vPairs, valPredsPtr, valDocs);
                    valNDCG = ComputeNDCG(valTargetsVec, valPredsPtr, valGroupOffsets, valNumGroups);
                } else {
                    valLoss = ComputeLossValue(valCursor, valTargetsArr, lossType, lossParam, approxDim, valDocs);
                }
                result.FinalTestLoss = valLoss;
                if (isRanking) {
                    printf("iter=%u  trees=%u  depth=%zu  loss=%.6f  NDCG=%.4f  val_loss=%.6f  val_NDCG=%.4f  time=%lldms\n",
                           iter, result.TreesBuilt, splits.size(), trainLoss, trainNDCG, valLoss, valNDCG, iterMs);
                } else {
                    printf("iter=%u  trees=%u  depth=%zu  train_loss=%.6f  val_loss=%.6f  time=%lldms\n",
                           iter, result.TreesBuilt, splits.size(), trainLoss, valLoss, iterMs);
                }

                // Early stopping check
                if (config.EarlyStoppingPatience > 0) {
                    if (valLoss < bestValLoss - 1e-7f) {
                        bestValLoss = valLoss;
                        bestIteration = iter;
                        noImprovementCount = 0;
                    } else {
                        noImprovementCount++;
                        if (noImprovementCount >= config.EarlyStoppingPatience) {
                            printf("Early stopping at iter=%u (best val_loss=%.6f at iter=%u)\n",
                                   iter, bestValLoss, bestIteration);
                            break;
                        }
                    }
                }
            } else {
                if (isRanking) {
                    printf("iter=%u  trees=%u  depth=%zu  loss=%.6f  NDCG=%.4f  time=%lldms\n",
                           iter, result.TreesBuilt, splits.size(), trainLoss, trainNDCG, iterMs);
                } else {
                    printf("iter=%u  trees=%u  depth=%zu  loss=%.6f  time=%lldms\n",
                           iter, result.TreesBuilt, splits.size(), trainLoss, iterMs);
                }
            }
        }

        // Track final loss even when not printing
        if (!printProgress || !(config.Verbose || iter % 10 == 0 || iter == config.NumIterations - 1)) {
            if (iter == config.NumIterations - 1 || (valDocs > 0 && noImprovementCount >= config.EarlyStoppingPatience)) {
                if (isRanking) {
                    auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
                    mx::eval(flatCursor);
                    const float* predsPtr = flatCursor.data<float>();
                    const auto& pairs = (lossType == "pairlogit") ? pairLogitPairs
                        : GenerateYetiRankPairs(trainTargetsVec, trainGroupOffsets, trainNumGroups, rng);
                    result.FinalTrainLoss = ComputePairLogitLoss(pairs, predsPtr, trainDocs);
                } else {
                    result.FinalTrainLoss = ComputeLossValue(cursor, targetsArr, lossType, lossParam, approxDim, trainDocs);
                }
                if (valDocs > 0) {
                    if (isRanking) {
                        auto flatVal = mx::reshape(valCursor, {static_cast<int>(valDocs)});
                        mx::eval(flatVal);
                        const float* valPredsPtr = flatVal.data<float>();
                        const auto& vPairs = (lossType == "pairlogit") ? valPairLogitPairs
                            : GenerateYetiRankPairs(valTargetsVec, valGroupOffsets, valNumGroups, rng);
                        result.FinalTestLoss = ComputePairLogitLoss(vPairs, valPredsPtr, valDocs);
                    } else {
                        result.FinalTestLoss = ComputeLossValue(valCursor, valTargetsArr, lossType, lossParam, approxDim, valDocs);
                    }
                }
            }
        }

        // Save snapshot at interval
        if (!snapshotPath.empty() && ((iter + 1) % snapshotInterval == 0 || iter == config.NumIterations - 1)) {
            SaveSnapshot(snapshotPath, iter, result.Trees, cursor, trainDocs,
                         valCursor, valDocs, approxDim,
                         bestValLoss, bestIteration, noImprovementCount, rng);
        }
    }

    result.BestIteration = bestIteration;
    return result;
}

// ============================================================================
// Cross-validation fold creation
// ============================================================================

std::vector<ui32> CreateStratifiedFolds(
    const std::vector<float>& targets, ui32 numFolds,
    const std::string& lossType, std::mt19937& rng
) {
    ui32 numDocs = targets.size();
    std::vector<ui32> foldAssignment(numDocs, 0);

    if (lossType == "logloss" || lossType == "multiclass") {
        // Stratified: group by class, distribute evenly
        std::unordered_map<int, std::vector<ui32>> classIndices;
        for (ui32 d = 0; d < numDocs; ++d) {
            classIndices[static_cast<int>(targets[d])].push_back(d);
        }
        for (auto& [cls, indices] : classIndices) {
            std::shuffle(indices.begin(), indices.end(), rng);
            for (ui32 i = 0; i < indices.size(); ++i) {
                foldAssignment[indices[i]] = i % numFolds;
            }
        }
    } else {
        // Random assignment for regression
        std::vector<ui32> indices(numDocs);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (ui32 i = 0; i < numDocs; ++i) {
            foldAssignment[indices[i]] = i % numFolds;
        }
    }
    return foldAssignment;
}

// ============================================================================
// Main training loop
// ============================================================================

int main(int argc, char** argv) {
    auto config = ParseArgs(argc, argv);

    printf("CatBoost-MLX CSV Training Tool\n");
    printf("==============================\n");

    // Parse loss type (may include parameter like "quantile:0.75" or "huber:1.5")
    auto lossConfig = ParseLossType(config.LossType);

    // Load data
    auto ds = LoadCSV(config.CsvPath, config.TargetCol, config.CatFeatureCols, config.NanMode, config.GroupCol, config.WeightCol);
    printf("Loaded: %u rows, %u features from %s\n", ds.NumDocs, ds.NumFeatures, config.CsvPath.c_str());

    // Report categorical features
    ui32 numCat = 0;
    for (ui32 f = 0; f < ds.NumFeatures; ++f) if (ds.IsCategorical[f]) numCat++;
    if (numCat > 0) {
        printf("Categorical features: %u (", numCat);
        bool first = true;
        for (ui32 f = 0; f < ds.NumFeatures; ++f) {
            if (ds.IsCategorical[f]) {
                if (!first) printf(", ");
                printf("f%u:%zu cats", f, ds.CatHashMaps[f].size());
                first = false;
            }
        }
        printf(")\n");
    }

    // Report NaN features
    ui32 numNaN = 0;
    for (ui32 f = 0; f < ds.NumFeatures; ++f) if (ds.HasNaN[f]) numNaN++;
    if (numNaN > 0) {
        printf("Features with NaN: %u\n", numNaN);
    }

    // Detect or validate loss type
    std::string lossType = lossConfig.Type;
    float lossParam = lossConfig.Param;
    if (lossType == "auto") {
        lossType = DetectLossType(ds.Targets);
        printf("Auto-detected loss: %s\n", lossType.c_str());
    }

    bool isRankingLoss = (lossType == "pairlogit" || lossType == "yetirank");
    if (isRankingLoss && ds.NumGroups == 0) {
        fprintf(stderr, "Error: Ranking losses (pairlogit, yetirank) require --group-col\n");
        return 1;
    }
    if (isRankingLoss) {
        printf("Ranking: %u groups, loss=%s\n", ds.NumGroups, lossType.c_str());
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

    // CTR target encoding for high-cardinality categoricals
    std::vector<TCtrFeature> ctrFeatures;
    if (config.UseCtr) {
        ctrFeatures = ComputeCtrFeatures(ds, lossType, numClasses,
                                          config.CtrPrior, config.MaxOneHotSize, config.RandomSeed);
        if (!ctrFeatures.empty()) {
            printf("CTR features: %zu (replaced high-cardinality categoricals with >%u values)\n",
                   ctrFeatures.size(), config.MaxOneHotSize);
        }
    }

    // Quantize and pack features
    auto quant = QuantizeFeatures(ds, config.MaxBins);
    auto packed = PackFeatures(quant, ds);
    printf("Quantized: %u bin-features, %u uint32 per doc\n",
           packed.TotalBinFeatures, packed.NumUi32PerDoc);
    if (packed.TotalBinFeatures == 0) {
        fprintf(stderr, "Error: No valid features after quantization\n");
        return 1;
    }

    // --- Cross-validation mode ---
    if (config.CVFolds > 1) {
        if (config.EvalFraction > 0.0f) {
            fprintf(stderr, "Error: --cv and --eval-fraction are mutually exclusive\n");
            return 1;
        }

        printf("\n%u-Fold Cross-Validation\n", config.CVFolds);
        printf("---\n");

        std::mt19937 cvRng(config.RandomSeed);

        // For ranking: fold assignment at group level
        std::vector<ui32> foldAssignment;
        if (isRankingLoss && ds.NumGroups > 0) {
            // Assign folds to groups, then expand to per-doc
            std::vector<ui32> groupFolds(ds.NumGroups);
            std::vector<ui32> groupPerm(ds.NumGroups);
            std::iota(groupPerm.begin(), groupPerm.end(), 0);
            std::shuffle(groupPerm.begin(), groupPerm.end(), cvRng);
            for (ui32 i = 0; i < ds.NumGroups; ++i) {
                groupFolds[groupPerm[i]] = i % config.CVFolds;
            }
            foldAssignment.resize(ds.NumDocs);
            for (ui32 g = 0; g < ds.NumGroups; ++g) {
                for (ui32 d = ds.GroupOffsets[g]; d < ds.GroupOffsets[g + 1]; ++d) {
                    foldAssignment[d] = groupFolds[g];
                }
            }
        } else {
            foldAssignment = CreateStratifiedFolds(ds.Targets, config.CVFolds, lossType, cvRng);
        }

        std::vector<float> foldTestLoss(config.CVFolds);
        std::vector<float> foldTrainLoss(config.CVFolds);

        for (ui32 fold = 0; fold < config.CVFolds; ++fold) {
            printf("\n=== Fold %u/%u ===\n", fold + 1, config.CVFolds);

            // Collect train/test indices
            std::vector<ui32> trainIdx, testIdx;
            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                if (foldAssignment[d] == fold) testIdx.push_back(d);
                else trainIdx.push_back(d);
            }

            ui32 foldTrainDocs = trainIdx.size();
            ui32 foldTestDocs = testIdx.size();
            printf("Train: %u docs, Test: %u docs\n", foldTrainDocs, foldTestDocs);

            // Build per-fold packed data from same quantization
            std::vector<uint32_t> trainData(foldTrainDocs * packed.NumUi32PerDoc);
            std::vector<uint32_t> testData(foldTestDocs * packed.NumUi32PerDoc);
            std::vector<float> trainTargets(foldTrainDocs);
            std::vector<float> testTargets(foldTestDocs);
            std::vector<float> foldTrainWeights;

            for (ui32 i = 0; i < foldTrainDocs; ++i) {
                ui32 d = trainIdx[i];
                for (ui32 w = 0; w < packed.NumUi32PerDoc; ++w)
                    trainData[i * packed.NumUi32PerDoc + w] = packed.Data[d * packed.NumUi32PerDoc + w];
                trainTargets[i] = ds.Targets[d];
            }
            if (!ds.Weights.empty()) {
                foldTrainWeights.resize(foldTrainDocs);
                for (ui32 i = 0; i < foldTrainDocs; ++i)
                    foldTrainWeights[i] = ds.Weights[trainIdx[i]];
            }
            for (ui32 i = 0; i < foldTestDocs; ++i) {
                ui32 d = testIdx[i];
                for (ui32 w = 0; w < packed.NumUi32PerDoc; ++w)
                    testData[i * packed.NumUi32PerDoc + w] = packed.Data[d * packed.NumUi32PerDoc + w];
                testTargets[i] = ds.Targets[d];
            }

            // Build group offsets for train/test subsets (ranking only)
            std::vector<ui32> foldTrainGroupOffsets, foldTestGroupOffsets;
            ui32 foldTrainNumGroups = 0, foldTestNumGroups = 0;
            if (isRankingLoss && ds.NumGroups > 0) {
                // Data within trainIdx/testIdx is already sorted by original group order
                // Rebuild group offsets from the group IDs
                auto buildGroupOffsets = [](const std::vector<ui32>& indices,
                                           const std::vector<ui32>& groupIds,
                                           std::vector<ui32>& offsets, ui32& numGroups) {
                    offsets.clear();
                    numGroups = 0;
                    if (indices.empty()) return;
                    offsets.push_back(0);
                    for (ui32 i = 1; i < indices.size(); ++i) {
                        if (groupIds[indices[i]] != groupIds[indices[i - 1]]) {
                            offsets.push_back(i);
                            numGroups++;
                        }
                    }
                    offsets.push_back(indices.size());
                    numGroups++;
                };
                buildGroupOffsets(trainIdx, ds.GroupIds, foldTrainGroupOffsets, foldTrainNumGroups);
                buildGroupOffsets(testIdx, ds.GroupIds, foldTestGroupOffsets, foldTestNumGroups);
            }

            auto foldCompressed = mx::array(
                reinterpret_cast<const int32_t*>(trainData.data()),
                {static_cast<int>(foldTrainDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
            auto foldTargets = mx::array(trainTargets.data(), {static_cast<int>(foldTrainDocs)}, mx::float32);

            auto foldTestCompressed = mx::array(
                reinterpret_cast<const int32_t*>(testData.data()),
                {static_cast<int>(foldTestDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
            auto foldTestTargets = mx::array(testTargets.data(), {static_cast<int>(foldTestDocs)}, mx::float32);
            mx::eval({foldCompressed, foldTargets, foldTestCompressed, foldTestTargets});

            auto foldResult = RunTraining(
                config, foldCompressed, foldTrainDocs, foldTargets,
                foldTestCompressed, foldTestDocs, foldTestTargets,
                packed, lossType, lossParam, approxDim, numClasses,
                config.Verbose,
                trainTargets, foldTrainGroupOffsets, foldTrainNumGroups,
                testTargets, foldTestGroupOffsets, foldTestNumGroups,
                foldTrainWeights
            );

            foldTrainLoss[fold] = foldResult.FinalTrainLoss;
            foldTestLoss[fold] = foldResult.FinalTestLoss;
            printf("Fold %u: train_loss=%.6f  test_loss=%.6f  trees=%u\n",
                   fold + 1, foldResult.FinalTrainLoss, foldResult.FinalTestLoss, foldResult.TreesBuilt);
        }

        // Aggregate results
        float meanTrain = 0, meanTest = 0;
        for (ui32 f = 0; f < config.CVFolds; ++f) {
            meanTrain += foldTrainLoss[f];
            meanTest += foldTestLoss[f];
        }
        meanTrain /= config.CVFolds;
        meanTest /= config.CVFolds;

        float stdTrain = 0, stdTest = 0;
        for (ui32 f = 0; f < config.CVFolds; ++f) {
            stdTrain += (foldTrainLoss[f] - meanTrain) * (foldTrainLoss[f] - meanTrain);
            stdTest += (foldTestLoss[f] - meanTest) * (foldTestLoss[f] - meanTest);
        }
        stdTrain = std::sqrt(stdTrain / config.CVFolds);
        stdTest = std::sqrt(stdTest / config.CVFolds);

        printf("\n===================================\n");
        printf("CV Results (%u folds):\n", config.CVFolds);
        printf("  Train loss: %.6f +/- %.6f\n", meanTrain, stdTrain);
        printf("  Test  loss: %.6f +/- %.6f\n", meanTest, stdTest);
        printf("===================================\n");

        return 0;
    }

    // --- Normal training mode ---
    ui32 trainDocs = ds.NumDocs;
    ui32 valDocs = 0;

    // For ranking, we need per-doc targets and group offsets for train/val subsets
    std::vector<float> trainTargetsVecForRanking;
    std::vector<ui32> trainGroupOffsetsForRanking;
    ui32 trainNumGroupsForRanking = 0;
    std::vector<float> valTargetsVecForRanking;
    std::vector<ui32> valGroupOffsetsForRanking;
    ui32 valNumGroupsForRanking = 0;

    // Mutual exclusivity check
    if (!config.EvalFile.empty() && config.EvalFraction > 0.0f) {
        fprintf(stderr, "Error: --eval-file and --eval-fraction are mutually exclusive\n");
        return 1;
    }

    // External eval file: load, quantize with training borders, pack
    TDataset evalDs;
    TPackedData evalPacked;
    if (!config.EvalFile.empty()) {
        evalDs = LoadCSV(config.EvalFile, config.TargetCol, config.CatFeatureCols,
                         config.NanMode, config.GroupCol, config.WeightCol);
        printf("Loaded eval data: %u rows, %u features from %s\n",
               evalDs.NumDocs, evalDs.NumFeatures, config.EvalFile.c_str());
        if (evalDs.NumFeatures != ds.NumFeatures) {
            fprintf(stderr, "Error: Eval data has %u features, training data has %u\n",
                    evalDs.NumFeatures, ds.NumFeatures);
            return 1;
        }
        auto evalQuant = QuantizeWithBorders(evalDs, quant, ds);
        evalPacked = PackFeatures(evalQuant, evalDs);
        valDocs = evalDs.NumDocs;
        // trainDocs stays as ds.NumDocs (use all training data)
        printf("Using external eval file: %u train, %u val\n", trainDocs, valDocs);
    }

    if (config.EvalFraction > 0.0f && config.EvalFraction < 1.0f) {
        if (isRankingLoss && ds.NumGroups > 0) {
            // Group-aware split: snap to group boundaries
            ui32 valGroups = static_cast<ui32>(ds.NumGroups * config.EvalFraction);
            if (valGroups == 0) valGroups = 1;
            ui32 trainGroups = ds.NumGroups - valGroups;
            if (trainGroups == 0) { trainGroups = 1; valGroups = ds.NumGroups - 1; }
            trainDocs = ds.GroupOffsets[trainGroups];
            valDocs = ds.NumDocs - trainDocs;
            trainNumGroupsForRanking = trainGroups;
            valNumGroupsForRanking = valGroups;
            // Build sub-group offsets
            trainGroupOffsetsForRanking.assign(ds.GroupOffsets.begin(), ds.GroupOffsets.begin() + trainGroups + 1);
            for (ui32 g = 0; g <= valGroups; ++g) {
                valGroupOffsetsForRanking.push_back(ds.GroupOffsets[trainGroups + g] - trainDocs);
            }
            printf("Validation split (group-aware): %u train (%u groups), %u val (%u groups)\n",
                   trainDocs, trainGroups, valDocs, valGroups);
        } else {
            valDocs = static_cast<ui32>(ds.NumDocs * config.EvalFraction);
            trainDocs = ds.NumDocs - valDocs;
            if (valDocs == 0 || trainDocs == 0) {
                fprintf(stderr, "Error: --eval-fraction too small or too large\n");
                return 1;
            }
            printf("Validation split: %u train, %u val (%.0f%%)\n", trainDocs, valDocs, config.EvalFraction * 100);
        }
    }

    // Build target vectors for ranking
    if (isRankingLoss) {
        if (!config.EvalFile.empty()) {
            // External eval file: use all training data for train, evalDs for val
            trainTargetsVecForRanking.assign(ds.Targets.begin(), ds.Targets.end());
            trainGroupOffsetsForRanking = ds.GroupOffsets;
            trainNumGroupsForRanking = ds.NumGroups;
            valTargetsVecForRanking.assign(evalDs.Targets.begin(), evalDs.Targets.end());
            valGroupOffsetsForRanking = evalDs.GroupOffsets;
            valNumGroupsForRanking = evalDs.NumGroups;
        } else {
            trainTargetsVecForRanking.assign(ds.Targets.begin(), ds.Targets.begin() + trainDocs);
            if (valDocs > 0) {
                valTargetsVecForRanking.assign(ds.Targets.begin() + trainDocs, ds.Targets.end());
            }
            // If no eval split, use full group offsets
            if (valDocs == 0) {
                trainGroupOffsetsForRanking = ds.GroupOffsets;
                trainNumGroupsForRanking = ds.NumGroups;
            }
        }
    }

    // Transfer to GPU — split train/val if needed
    mx::array compressedData = mx::array(0);
    mx::array targetsArr = mx::array(0.0f);
    mx::array valCompressedData = mx::array(0);
    mx::array valTargetsArr = mx::array(0.0f);

    if (!config.EvalFile.empty()) {
        // External eval file: training data = all of ds, val data = evalDs (separately packed)
        compressedData = mx::array(
            reinterpret_cast<const int32_t*>(packed.Data.data()),
            {static_cast<int>(trainDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        targetsArr = mx::array(ds.Targets.data(), {static_cast<int>(trainDocs)}, mx::float32);

        valCompressedData = mx::array(
            reinterpret_cast<const int32_t*>(evalPacked.Data.data()),
            {static_cast<int>(valDocs), static_cast<int>(evalPacked.NumUi32PerDoc)}, mx::uint32);
        std::vector<float> evalTargetsVec(evalDs.Targets.begin(), evalDs.Targets.end());
        valTargetsArr = mx::array(evalTargetsVec.data(), {static_cast<int>(valDocs)}, mx::float32);
        mx::eval({compressedData, targetsArr, valCompressedData, valTargetsArr});
    } else if (valDocs > 0) {
        std::vector<uint32_t> trainData(trainDocs * packed.NumUi32PerDoc);
        std::vector<uint32_t> valData(valDocs * packed.NumUi32PerDoc);
        std::vector<float> trainTargetsVec(trainDocs);
        std::vector<float> valTargetsVec(valDocs);

        for (ui32 d = 0; d < trainDocs; ++d) {
            for (ui32 w = 0; w < packed.NumUi32PerDoc; ++w)
                trainData[d * packed.NumUi32PerDoc + w] = packed.Data[d * packed.NumUi32PerDoc + w];
            trainTargetsVec[d] = ds.Targets[d];
        }
        for (ui32 d = 0; d < valDocs; ++d) {
            for (ui32 w = 0; w < packed.NumUi32PerDoc; ++w)
                valData[d * packed.NumUi32PerDoc + w] = packed.Data[(trainDocs + d) * packed.NumUi32PerDoc + w];
            valTargetsVec[d] = ds.Targets[trainDocs + d];
        }

        compressedData = mx::array(
            reinterpret_cast<const int32_t*>(trainData.data()),
            {static_cast<int>(trainDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        targetsArr = mx::array(trainTargetsVec.data(), {static_cast<int>(trainDocs)}, mx::float32);

        valCompressedData = mx::array(
            reinterpret_cast<const int32_t*>(valData.data()),
            {static_cast<int>(valDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        valTargetsArr = mx::array(valTargetsVec.data(), {static_cast<int>(valDocs)}, mx::float32);
        mx::eval({compressedData, targetsArr, valCompressedData, valTargetsArr});
    } else {
        compressedData = mx::array(
            reinterpret_cast<const int32_t*>(packed.Data.data()),
            {static_cast<int>(ds.NumDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        targetsArr = mx::array(ds.Targets.data(), {static_cast<int>(ds.NumDocs)}, mx::float32);
        mx::eval({compressedData, targetsArr});
    }

    // Print training configuration
    std::string lossDisplay = lossType;
    if (lossType == "quantile") lossDisplay += ":" + std::to_string(lossParam);
    if (lossType == "huber") lossDisplay += ":" + std::to_string(lossParam);
    if (lossType == "tweedie") lossDisplay += ":" + std::to_string(lossParam);
    printf("\nTraining: %u iterations, depth=%u, lr=%.4f, l2=%.2f, loss=%s\n",
           config.NumIterations, config.MaxDepth, config.LearningRate, config.L2RegLambda, lossDisplay.c_str());
    if (config.SubsampleRatio < 1.0f) printf("Row subsample: %.2f\n", config.SubsampleRatio);
    if (config.BootstrapType != "no") printf("Bootstrap: %s", config.BootstrapType.c_str());
    if (config.BootstrapType == "bayesian") printf(" (temperature=%.2f)", config.BaggingTemperature);
    if (config.BootstrapType == "mvs") printf(" (reg=%.2f)", config.MvsReg);
    if (config.BootstrapType != "no") printf("\n");
    if (config.ColsampleByTree < 1.0f) printf("Col subsample: %.2f\n", config.ColsampleByTree);
    if (config.WeightCol >= 0) printf("Sample weight column: %d\n", config.WeightCol);
    if (config.MinDataInLeaf > 1) printf("Min data in leaf: %u\n", config.MinDataInLeaf);
    if (!config.MonotoneConstraints.empty()) {
        printf("Monotone constraints:");
        for (auto c : config.MonotoneConstraints) printf(" %d", c);
        printf("\n");
    }
    if (config.EarlyStoppingPatience > 0) printf("Early stopping patience: %u\n", config.EarlyStoppingPatience);
    printf("---\n");

    auto startTime = std::chrono::steady_clock::now();

    // Slice sample weights for training subset only
    std::vector<float> trainWeights;
    if (!ds.Weights.empty()) {
        trainWeights.assign(ds.Weights.begin(), ds.Weights.begin() + trainDocs);
    }

    auto trainResult = RunTraining(
        config, compressedData, trainDocs, targetsArr,
        valCompressedData, valDocs, valTargetsArr,
        packed, lossType, lossParam, approxDim, numClasses,
        true,  // print progress
        trainTargetsVecForRanking, trainGroupOffsetsForRanking, trainNumGroupsForRanking,
        valTargetsVecForRanking, valGroupOffsetsForRanking, valNumGroupsForRanking,
        trainWeights,
        config.SnapshotPath, config.SnapshotInterval
    );

    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count();

    printf("---\n");
    printf("Training complete: %u trees in %.2fs\n", trainResult.TreesBuilt, totalTime / 1000.0);

    // Feature importance
    if (config.ShowFeatureImportance && !trainResult.Trees.empty()) {
        std::vector<double> featureGain(ds.NumFeatures, 0.0);
        for (const auto& tree : trainResult.Trees) {
            for (ui32 level = 0; level < tree.SplitProps.size(); ++level) {
                ui32 featIdx = tree.SplitProps[level].FeatureId;
                if (featIdx < ds.NumFeatures) {
                    featureGain[featIdx] += tree.SplitProps[level].Gain;
                }
            }
        }
        double totalGain = 0.0;
        for (double g : featureGain) totalGain += g;

        std::vector<ui32> sortedIndices(ds.NumFeatures);
        std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                  [&](ui32 a, ui32 b) { return featureGain[a] > featureGain[b]; });

        printf("\nFeature Importance (Gain-based):\n");
        printf("%-4s  %-20s  %10s  %6s\n", "Rank", "Feature", "Gain", "%");
        printf("----  --------------------  ----------  ------\n");
        for (ui32 rank = 0; rank < ds.NumFeatures; ++rank) {
            ui32 fi = sortedIndices[rank];
            if (featureGain[fi] <= 0.0) continue;
            std::string name = (fi < ds.FeatureNames.size() && !ds.FeatureNames[fi].empty())
                ? ds.FeatureNames[fi] : ("f" + std::to_string(fi));
            double pct = (totalGain > 0) ? 100.0 * featureGain[fi] / totalGain : 0.0;
            printf("%-4u  %-20s  %10.4f  %5.1f%%\n", rank + 1, name.c_str(), featureGain[fi], pct);
        }
    }

    // Save model
    if (!config.OutputModelPath.empty() && !trainResult.Trees.empty()) {
        SaveModelJSON(config.OutputModelPath, trainResult.Trees, ds, quant, lossType, lossParam,
                      config, approxDim, numClasses, ctrFeatures);
        printf("Model saved to: %s (%u trees, %u features)\n",
               config.OutputModelPath.c_str(), static_cast<ui32>(trainResult.Trees.size()), ds.NumFeatures);
    }

    return 0;
}
