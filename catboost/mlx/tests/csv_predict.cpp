// CatBoost-MLX standalone CSV prediction tool.
// Loads a JSON model (saved by csv_train --output) and applies it to new CSV data.
//
// Usage: ./csv_predict <model.json> <data.csv> [options]
//   --output PATH          Write predictions to CSV file (default: stdout)
//   --target-col N         0-based column index for target (for evaluation; default: -1 = no target)
//   --verbose              Print detailed info
//
// Compile:
//   clang++ -std=c++17 -O2 -I. \
//     -I/opt/homebrew/Cellar/mlx/0.31.1/include \
//     -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
//     -framework Metal -framework Foundation -Wno-c++20-extensions \
//     catboost/mlx/tests/csv_predict.cpp -o csv_predict

#include <mlx/mlx.h>

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <unordered_map>

namespace mx = mlx::core;
using ui32 = uint32_t;

// ============================================================================
// Structures (mirrored from csv_train.cpp for standalone compilation)
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

struct TObliviousSplitLevel {
    ui32 FeatureColumnIdx;
    ui32 Shift;
    ui32 Mask;
    ui32 BinThreshold;
    bool IsOneHot = false;
};

// ============================================================================
// CLI argument parsing
// ============================================================================

struct TPredictConfig {
    std::string ModelPath;
    std::string CsvPath;
    std::string OutputPath;  // "" = stdout
    int TargetCol = -1;      // -1 = no target column
    int GroupCol = -1;       // -1 = no group column (ranking only)
    bool Verbose = false;
    bool ComputeShap = false;
};

TPredictConfig ParseArgs(int argc, char** argv) {
    TPredictConfig config;
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.json> <data.csv> [--output PATH] [--target-col N] [--group-col N] [--shap] [--verbose]\n", argv[0]);
        exit(1);
    }
    config.ModelPath = argv[1];
    config.CsvPath = argv[2];
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) config.OutputPath = argv[++i];
        else if (strcmp(argv[i], "--target-col") == 0 && i + 1 < argc) config.TargetCol = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--group-col") == 0 && i + 1 < argc) config.GroupCol = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0) config.Verbose = true;
        else if (strcmp(argv[i], "--shap") == 0) config.ComputeShap = true;
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); exit(1); }
    }
    return config;
}

// ============================================================================
// JSON Model loading (minimal hand-written parser for our fixed schema)
// ============================================================================

struct TModelFeature {
    ui32 Index;
    std::string Name;
    bool IsCategorical;
    bool HasNaN;
    std::vector<float> Borders;
    std::unordered_map<std::string, uint32_t> CatHashMap;
};

struct TModelSplit {
    ui32 FeatureIdx;
    ui32 BinThreshold;
    bool IsOneHot;
};

struct TModelTree {
    ui32 Depth;
    std::vector<TModelSplit> Splits;
    std::vector<float> LeafValues;
    std::vector<float> SplitGains;
};

struct TModelInfo {
    std::string LossType;
    float LossParam;
    float LearningRate;
    float L2RegLambda;
    ui32 ApproxDimension;
    ui32 NumClasses;
    ui32 NumTrees;
    ui32 MaxDepth;
    std::string NanMode;
    std::vector<float> BasePrediction;  // optimal starting constant per dimension
};

struct TModel {
    TModelInfo Info;
    std::vector<TModelFeature> Features;
    std::vector<TModelTree> Trees;
    // CTR features (empty if no CTR was used)
    struct TCtrInfo {
        ui32 OriginalFeatureIdx;
        std::string Name;
        float Prior;
        ui32 ClassIdx;
        float DefaultValue;
        std::unordered_map<uint32_t, float> FinalValues;  // catBin → CTR value
        std::unordered_map<std::string, uint32_t> CatHashMap;  // string → catBin
    };
    std::vector<TCtrInfo> CtrFeatures;
};

// Minimal JSON tokenizer
struct TJsonParser {
    const std::string& data;
    size_t pos = 0;

    TJsonParser(const std::string& s) : data(s), pos(0) {}

    void SkipWhitespace() {
        while (pos < data.size() && (data[pos] == ' ' || data[pos] == '\t' || data[pos] == '\n' || data[pos] == '\r'))
            pos++;
    }

    char Peek() { SkipWhitespace(); return pos < data.size() ? data[pos] : '\0'; }
    char Next() { SkipWhitespace(); return pos < data.size() ? data[pos++] : '\0'; }

    void Expect(char c) {
        char got = Next();
        if (got != c) {
            fprintf(stderr, "JSON parse error at pos %zu: expected '%c', got '%c'\n", pos, c, got);
            exit(1);
        }
    }

    std::string ParseString() {
        Expect('"');
        std::string result;
        while (pos < data.size() && data[pos] != '"') {
            if (data[pos] == '\\') {
                pos++;
                if (pos < data.size()) {
                    switch (data[pos]) {
                        case '"':  result += '"'; break;
                        case '\\': result += '\\'; break;
                        case 'n':  result += '\n'; break;
                        case 'r':  result += '\r'; break;
                        case 't':  result += '\t'; break;
                        default:   result += data[pos]; break;
                    }
                }
            } else {
                result += data[pos];
            }
            pos++;
        }
        Expect('"');
        return result;
    }

    double ParseNumber() {
        SkipWhitespace();
        size_t start = pos;
        if (pos < data.size() && (data[pos] == '-' || data[pos] == '+')) pos++;
        while (pos < data.size() && (std::isdigit(data[pos]) || data[pos] == '.' || data[pos] == 'e' || data[pos] == 'E' || data[pos] == '+' || data[pos] == '-')) {
            // Handle signs that appear after 'e'/'E'
            if ((data[pos] == '+' || data[pos] == '-') && pos > start && data[pos-1] != 'e' && data[pos-1] != 'E') break;
            pos++;
        }
        return std::stod(data.substr(start, pos - start));
    }

    bool ParseBool() {
        SkipWhitespace();
        if (data.substr(pos, 4) == "true") { pos += 4; return true; }
        if (data.substr(pos, 5) == "false") { pos += 5; return false; }
        fprintf(stderr, "JSON parse error at pos %zu: expected bool\n", pos);
        exit(1);
    }

    void SkipValue() {
        SkipWhitespace();
        if (data[pos] == '"') { ParseString(); return; }
        if (data[pos] == '{') {
            Next(); // {
            if (Peek() != '}') {
                do {
                    ParseString(); Expect(':'); SkipValue();
                } while (Peek() == ',' && (Next(), true));
            }
            Expect('}');
            return;
        }
        if (data[pos] == '[') {
            Next(); // [
            if (Peek() != ']') {
                do { SkipValue(); } while (Peek() == ',' && (Next(), true));
            }
            Expect(']');
            return;
        }
        if (data.substr(pos, 4) == "true") { pos += 4; return; }
        if (data.substr(pos, 5) == "false") { pos += 5; return; }
        if (data.substr(pos, 4) == "null") { pos += 4; return; }
        // number
        ParseNumber();
    }

    // Parse "key": and return key. Caller must handle the value.
    std::string ParseKey() {
        auto key = ParseString();
        Expect(':');
        return key;
    }
};

TModel LoadModelJSON(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open model file: %s\n", path.c_str());
        exit(1);
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    TJsonParser p(content);
    TModel model;

    p.Expect('{');
    while (p.Peek() != '}') {
        auto key = p.ParseKey();

        if (key == "format" || key == "version") {
            p.SkipValue();
        } else if (key == "model_info") {
            p.Expect('{');
            while (p.Peek() != '}') {
                auto k = p.ParseKey();
                if (k == "loss_type") model.Info.LossType = p.ParseString();
                else if (k == "loss_param") model.Info.LossParam = static_cast<float>(p.ParseNumber());
                else if (k == "learning_rate") model.Info.LearningRate = static_cast<float>(p.ParseNumber());
                else if (k == "l2_reg_lambda") model.Info.L2RegLambda = static_cast<float>(p.ParseNumber());
                else if (k == "approx_dimension") model.Info.ApproxDimension = static_cast<ui32>(p.ParseNumber());
                else if (k == "num_classes") model.Info.NumClasses = static_cast<ui32>(p.ParseNumber());
                else if (k == "num_trees") model.Info.NumTrees = static_cast<ui32>(p.ParseNumber());
                else if (k == "max_depth") model.Info.MaxDepth = static_cast<ui32>(p.ParseNumber());
                else if (k == "nan_mode") model.Info.NanMode = p.ParseString();
                else if (k == "base_prediction") {
                    p.Expect('[');
                    while (p.Peek() != ']') {
                        model.Info.BasePrediction.push_back(static_cast<float>(p.ParseNumber()));
                        if (p.Peek() == ',') p.Next();
                    }
                    p.Expect(']');
                }
                else p.SkipValue();
                if (p.Peek() == ',') p.Next();
            }
            p.Expect('}');
        } else if (key == "features") {
            p.Expect('[');
            while (p.Peek() != ']') {
                TModelFeature feat;
                p.Expect('{');
                while (p.Peek() != '}') {
                    auto k = p.ParseKey();
                    if (k == "index") feat.Index = static_cast<ui32>(p.ParseNumber());
                    else if (k == "name") feat.Name = p.ParseString();
                    else if (k == "is_categorical") feat.IsCategorical = p.ParseBool();
                    else if (k == "has_nan") feat.HasNaN = p.ParseBool();
                    else if (k == "borders") {
                        p.Expect('[');
                        while (p.Peek() != ']') {
                            feat.Borders.push_back(static_cast<float>(p.ParseNumber()));
                            if (p.Peek() == ',') p.Next();
                        }
                        p.Expect(']');
                    } else if (k == "cat_hash_map") {
                        p.Expect('{');
                        while (p.Peek() != '}') {
                            auto catKey = p.ParseString();
                            p.Expect(':');
                            auto catVal = static_cast<uint32_t>(p.ParseNumber());
                            feat.CatHashMap[catKey] = catVal;
                            if (p.Peek() == ',') p.Next();
                        }
                        p.Expect('}');
                    } else p.SkipValue();
                    if (p.Peek() == ',') p.Next();
                }
                p.Expect('}');
                model.Features.push_back(std::move(feat));
                if (p.Peek() == ',') p.Next();
            }
            p.Expect(']');
        } else if (key == "trees") {
            p.Expect('[');
            while (p.Peek() != ']') {
                TModelTree tree;
                p.Expect('{');
                while (p.Peek() != '}') {
                    auto k = p.ParseKey();
                    if (k == "depth") tree.Depth = static_cast<ui32>(p.ParseNumber());
                    else if (k == "splits") {
                        p.Expect('[');
                        while (p.Peek() != ']') {
                            TModelSplit split;
                            p.Expect('{');
                            while (p.Peek() != '}') {
                                auto sk = p.ParseKey();
                                if (sk == "feature_idx") split.FeatureIdx = static_cast<ui32>(p.ParseNumber());
                                else if (sk == "bin_threshold") split.BinThreshold = static_cast<ui32>(p.ParseNumber());
                                else if (sk == "is_one_hot") split.IsOneHot = p.ParseBool();
                                else p.SkipValue();
                                if (p.Peek() == ',') p.Next();
                            }
                            p.Expect('}');
                            tree.Splits.push_back(split);
                            if (p.Peek() == ',') p.Next();
                        }
                        p.Expect(']');
                    } else if (k == "leaf_values") {
                        p.Expect('[');
                        while (p.Peek() != ']') {
                            tree.LeafValues.push_back(static_cast<float>(p.ParseNumber()));
                            if (p.Peek() == ',') p.Next();
                        }
                        p.Expect(']');
                    } else if (k == "split_gains") {
                        p.Expect('[');
                        while (p.Peek() != ']') {
                            tree.SplitGains.push_back(static_cast<float>(p.ParseNumber()));
                            if (p.Peek() == ',') p.Next();
                        }
                        p.Expect(']');
                    } else p.SkipValue();
                    if (p.Peek() == ',') p.Next();
                }
                p.Expect('}');
                model.Trees.push_back(std::move(tree));
                if (p.Peek() == ',') p.Next();
            }
            p.Expect(']');
        } else if (key == "ctr_features") {
            p.Expect('[');
            while (p.Peek() != ']') {
                TModel::TCtrInfo ctr;
                p.Expect('{');
                while (p.Peek() != '}') {
                    auto k = p.ParseKey();
                    if (k == "original_feature_idx") ctr.OriginalFeatureIdx = static_cast<ui32>(p.ParseNumber());
                    else if (k == "name") ctr.Name = p.ParseString();
                    else if (k == "prior") ctr.Prior = static_cast<float>(p.ParseNumber());
                    else if (k == "class_idx") ctr.ClassIdx = static_cast<ui32>(p.ParseNumber());
                    else if (k == "default_value") ctr.DefaultValue = static_cast<float>(p.ParseNumber());
                    else if (k == "final_values") {
                        p.Expect('{');
                        while (p.Peek() != '}') {
                            auto binStr = p.ParseString();
                            p.Expect(':');
                            float val = static_cast<float>(p.ParseNumber());
                            ctr.FinalValues[static_cast<uint32_t>(std::stoul(binStr))] = val;
                            if (p.Peek() == ',') p.Next();
                        }
                        p.Expect('}');
                    } else if (k == "cat_hash_map") {
                        p.Expect('{');
                        while (p.Peek() != '}') {
                            auto catStr = p.ParseString();
                            p.Expect(':');
                            auto catBin = static_cast<uint32_t>(p.ParseNumber());
                            ctr.CatHashMap[catStr] = catBin;
                            if (p.Peek() == ',') p.Next();
                        }
                        p.Expect('}');
                    } else p.SkipValue();
                    if (p.Peek() == ',') p.Next();
                }
                p.Expect('}');
                model.CtrFeatures.push_back(std::move(ctr));
                if (p.Peek() == ',') p.Next();
            }
            p.Expect(']');
        } else {
            p.SkipValue();
        }
        if (p.Peek() == ',') p.Next();
    }
    p.Expect('}');

    return model;
}

// ============================================================================
// CSV loading for prediction (no target required)
// ============================================================================

bool IsNaNString(const std::string& s) {
    return s.empty() || s == "NaN" || s == "nan" || s == "NA" || s == "na" || s == "N/A" || s == "?";
}

struct TPredictDataset {
    std::vector<std::vector<std::string>> RawFeatures;  // [numFeatures][numDocs] raw strings
    std::vector<float> Targets;                          // [numDocs] (empty if no target col)
    ui32 NumDocs = 0;
    ui32 NumFeatures = 0;
    std::vector<std::string> FeatureNames;
    bool HasTargets = false;
};

TPredictDataset LoadPredictCSV(const std::string& path, int targetCol, int groupCol = -1) {
    TPredictDataset ds;
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file: %s\n", path.c_str());
        exit(1);
    }

    std::string line;
    std::vector<std::vector<std::string>> rawRows;
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
            // Check if first row is a header
            bool allNonNumeric = true;
            for (const auto& c : cells) {
                char* end = nullptr;
                std::strtod(c.c_str(), &end);
                if (end != c.c_str() && *end == '\0') { allNonNumeric = false; break; }
            }
            if (allNonNumeric && !cells.empty()) {
                hasHeader = true;
                ds.FeatureNames = cells;
                continue;
            }
        }
        if (static_cast<int>(cells.size()) == numCols) rawRows.push_back(cells);
    }

    ds.NumDocs = rawRows.size();
    if (ds.NumDocs == 0) {
        fprintf(stderr, "Error: No data rows found in %s\n", path.c_str());
        exit(1);
    }

    // Determine target column
    ds.HasTargets = (targetCol >= 0 && targetCol < numCols);
    ui32 excludedCols = (ds.HasTargets ? 1 : 0) + (groupCol >= 0 ? 1 : 0);
    ds.NumFeatures = numCols - excludedCols;

    // Build feature columns (excluding target and group)
    ds.RawFeatures.resize(ds.NumFeatures);
    for (ui32 f = 0; f < ds.NumFeatures; ++f) {
        ds.RawFeatures[f].resize(ds.NumDocs);
    }

    // Build feature name mapping
    if (hasHeader) {
        std::vector<std::string> featureNames;
        for (int c = 0; c < numCols; ++c) {
            if (c == targetCol) continue;
            if (c == groupCol) continue;
            featureNames.push_back(ds.FeatureNames[c]);
        }
        ds.FeatureNames = featureNames;
    }

    for (ui32 d = 0; d < ds.NumDocs; ++d) {
        ui32 fi = 0;
        for (int c = 0; c < numCols; ++c) {
            if (c == targetCol) {
                if (ds.HasTargets) ds.Targets.push_back(std::stof(rawRows[d][c]));
                continue;
            }
            if (c == groupCol) continue;
            ds.RawFeatures[fi][d] = rawRows[d][c];
            fi++;
        }
    }

    return ds;
}

// ============================================================================
// Quantize prediction data using model's borders and hash maps
// ============================================================================

struct TPackedData {
    std::vector<uint32_t> Data;  // [numDocs * numUi32PerDoc]
    std::vector<TCFeature> Features;
    ui32 NumUi32PerDoc;
    ui32 TotalBinFeatures;
};

TPackedData QuantizeAndPack(
    const TPredictDataset& ds,
    const TModel& model,
    bool verbose
) {
    const ui32 numDocs = ds.NumDocs;
    const ui32 numFeatures = model.Features.size();

    // First quantize all features into bin indices
    std::vector<std::vector<uint8_t>> binnedFeatures(numFeatures);

    for (ui32 f = 0; f < numFeatures; ++f) {
        const auto& mf = model.Features[f];
        binnedFeatures[f].resize(numDocs, 0);

        if (mf.IsCategorical) {
            ui32 unknownCount = 0;
            for (ui32 d = 0; d < numDocs; ++d) {
                const auto& val = ds.RawFeatures[f][d];
                auto it = mf.CatHashMap.find(val);
                if (it != mf.CatHashMap.end()) {
                    binnedFeatures[f][d] = static_cast<uint8_t>(it->second);
                } else {
                    binnedFeatures[f][d] = 0;  // unknown category → bin 0
                    unknownCount++;
                }
            }
            if (verbose && unknownCount > 0) {
                printf("Warning: feature '%s' has %u unknown categories (mapped to bin 0)\n",
                       mf.Name.c_str(), unknownCount);
            }
        } else {
            // Numeric: apply borders with NaN handling
            bool hasNaN = mf.HasNaN;
            ui32 binOffset = hasNaN ? 1 : 0;

            for (ui32 d = 0; d < numDocs; ++d) {
                const auto& val = ds.RawFeatures[f][d];
                if (IsNaNString(val)) {
                    binnedFeatures[f][d] = 0;  // NaN → bin 0
                } else {
                    float fval = std::stof(val);
                    if (std::isnan(fval)) {
                        binnedFeatures[f][d] = 0;
                    } else {
                        auto it = std::upper_bound(mf.Borders.begin(), mf.Borders.end(), fval);
                        binnedFeatures[f][d] = static_cast<uint8_t>((it - mf.Borders.begin()) + binOffset);
                    }
                }
            }
        }
    }

    // Pack into compressed uint32 format (identical to csv_train.cpp PackFeatures)
    TPackedData packed;
    packed.NumUi32PerDoc = (numFeatures + 3) / 4;
    packed.Data.resize(numDocs * packed.NumUi32PerDoc, 0);
    packed.TotalBinFeatures = 0;

    for (ui32 f = 0; f < numFeatures; ++f) {
        const auto& mf = model.Features[f];
        ui32 wordIdx = f / 4;
        ui32 posInWord = f % 4;
        ui32 shift = (3 - posInWord) * 8;
        ui32 mask = 0xFF << shift;

        ui32 folds;
        if (mf.IsCategorical) {
            folds = static_cast<ui32>(mf.CatHashMap.size());
        } else {
            folds = static_cast<ui32>(mf.Borders.size());
            if (mf.HasNaN) folds += 1;
        }

        TCFeature feat;
        feat.Offset = wordIdx;
        feat.Mask = mask;
        feat.Shift = shift;
        feat.FirstFoldIndex = packed.TotalBinFeatures;
        feat.Folds = folds;
        feat.OneHotFeature = mf.IsCategorical;
        feat.SkipFirstBinInScoreCount = false;
        packed.Features.push_back(feat);
        packed.TotalBinFeatures += folds;

        for (ui32 d = 0; d < numDocs; ++d) {
            packed.Data[d * packed.NumUi32PerDoc + wordIdx] |=
                (static_cast<uint32_t>(binnedFeatures[f][d]) << shift);
        }
    }

    return packed;
}

// ============================================================================
// Apply tree splits to compute leaf indices (duplicated from csv_train.cpp)
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
// Softmax computation (duplicated from csv_train.cpp)
// ============================================================================

struct TSoftmaxResult {
    mx::array Probs;  // [K, numDocs]
};

TSoftmaxResult ComputeSoftmax(const mx::array& cursor, int K, int numDocs) {
    auto maxC = mx::maximum(mx::max(cursor, 0), mx::array(0.0f));
    auto expC = mx::exp(mx::subtract(cursor, mx::reshape(maxC, {1, numDocs})));
    auto expImp = mx::exp(mx::negative(maxC));
    auto sumExp = mx::add(mx::sum(expC, 0), expImp);
    auto probs = mx::divide(expC, mx::reshape(sumExp, {1, numDocs}));
    return {probs};
}

// ============================================================================
// Build TObliviousSplitLevel from model splits + packed feature info
// ============================================================================

std::vector<TObliviousSplitLevel> BuildSplitLevels(
    const TModelTree& tree,
    const TPackedData& packed
) {
    std::vector<TObliviousSplitLevel> splits;
    splits.reserve(tree.Splits.size());

    for (const auto& ms : tree.Splits) {
        const auto& feat = packed.Features[ms.FeatureIdx];
        TObliviousSplitLevel split;
        split.FeatureColumnIdx = static_cast<ui32>(feat.Offset);
        split.Shift = feat.Shift;
        split.Mask = feat.Mask >> feat.Shift;
        split.BinThreshold = ms.BinThreshold;
        split.IsOneHot = ms.IsOneHot;
        splits.push_back(split);
    }
    return splits;
}

// ============================================================================
// ============================================================================
// TreeSHAP for oblivious (symmetric) trees
// ============================================================================
//
// For oblivious trees, all nodes at depth i use the same feature and threshold.
// A sample's leaf index is a D-bit integer: bit i = 1 means "went right at level i".
//
// The algorithm computes SHAP values per tree per sample using the Lundberg
// "Tree SHAP" approach, adapted for the oblivious structure where each depth
// level maps to exactly one feature.
//
// Complexity: O(D * 2^D) per tree per sample, where D = tree depth.
// For D=6: 384 operations per tree per sample.
//
// References:
//   - Lundberg, Lee (2017): "A Unified Approach to Interpreting Model Predictions"
//   - Lundberg et al. (2020): "From local explanations to global understanding
//     with explainable AI for trees"

// Compute the fraction of background data going right at each level of a tree.
// This is needed for the TreeSHAP "cover" (the fraction of data at each node
// that goes in each direction).
static std::vector<float> ComputeLevelRightFractions(
    const TModelTree& tree,
    const TPackedData& packed,
    const std::vector<uint32_t>& compressedDataFlat,
    ui32 numDocs
) {
    ui32 depth = tree.Depth;
    std::vector<float> rightFrac(depth, 0.0f);

    for (ui32 level = 0; level < depth; ++level) {
        const auto& split = tree.Splits[level];
        ui32 featureIdx = split.FeatureIdx;
        ui32 binThreshold = split.BinThreshold;
        bool isOneHot = split.IsOneHot;

        if (featureIdx >= packed.Features.size()) continue;
        const auto& feat = packed.Features[featureIdx];
        ui32 rawMask = feat.Mask >> feat.Shift;  // get unshifted mask (0xFF)

        ui32 rightCount = 0;
        for (ui32 d = 0; d < numDocs; ++d) {
            uint32_t word = compressedDataFlat[d * packed.NumUi32PerDoc + feat.Offset];
            uint32_t binVal = (word >> feat.Shift) & rawMask;
            bool goRight = isOneHot ? (binVal == binThreshold) : (binVal > binThreshold);
            if (goRight) rightCount++;
        }
        rightFrac[level] = static_cast<float>(rightCount) / static_cast<float>(numDocs);
    }
    return rightFrac;
}

// TreeSHAP for a single oblivious tree and a single sample.
// Returns SHAP values indexed by model feature index.
//
// For oblivious trees, the SHAP value for feature f is:
//   phi_f = sum over subsets S (not containing f) of:
//     weight(S) * [E[leaf | S union {f}] - E[leaf | S]]
//
// We use a direct enumeration approach over the 2^D leaf nodes.
// For each leaf, we compute the product of "path probabilities" for all
// possible feature subsets, weighting by the combinatorial SHAP weights.
static void ComputeTreeShapForDoc(
    const TModelTree& tree,
    const std::vector<float>& rightFrac,
    const std::vector<ui32>& levelToFeature,  // level → model feature index
    ui32 docLeafIndex,  // which leaf this doc falls into
    ui32 numModelFeatures,
    ui32 approxDim,
    std::vector<std::vector<float>>& shapValues  // output: [numFeatures][approxDim], accumulated
) {
    ui32 depth = tree.Depth;
    ui32 numLeaves = 1u << depth;

    // Precompute: for each feature, which levels it controls
    std::unordered_map<ui32, std::vector<ui32>> featureLevels;
    for (ui32 level = 0; level < depth; ++level) {
        featureLevels[levelToFeature[level]].push_back(level);
    }

    // Get unique features in this tree
    std::vector<ui32> uniqueFeatures;
    for (const auto& [feat, _] : featureLevels) {
        uniqueFeatures.push_back(feat);
    }
    ui32 M = uniqueFeatures.size();

    // Precompute factorials for SHAP weights
    std::vector<double> factorials(M + 1, 1.0);
    for (ui32 i = 1; i <= M; ++i) factorials[i] = factorials[i - 1] * i;

    for (ui32 fi = 0; fi < M; ++fi) {
        ui32 targetFeature = uniqueFeatures[fi];

        // Collect the other features
        std::vector<ui32> otherFeatures;
        for (ui32 fj = 0; fj < M; ++fj) {
            if (fj != fi) otherFeatures.push_back(uniqueFeatures[fj]);
        }
        ui32 numOther = otherFeatures.size();
        ui32 numSubsets = 1u << numOther;

        for (ui32 smask = 0; smask < numSubsets; ++smask) {
            ui32 sSize = __builtin_popcount(smask);
            double weight = factorials[sSize] * factorials[M - sSize - 1] / factorials[M];

            // Compute E[v | S] and E[v | S + targetFeature] per dimension
            std::vector<float> ev_with(approxDim, 0.0f);
            std::vector<float> ev_without(approxDim, 0.0f);

            for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
                float pathProb_with = 1.0f;
                float pathProb_without = 1.0f;

                for (ui32 level = 0; level < depth; ++level) {
                    ui32 levelFeat = levelToFeature[level];
                    bool leafGoesRight = (leaf >> level) & 1;

                    bool inS = false;
                    if (levelFeat != targetFeature) {
                        for (ui32 oi = 0; oi < numOther; ++oi) {
                            if (otherFeatures[oi] == levelFeat && ((smask >> oi) & 1)) {
                                inS = true;
                                break;
                            }
                        }
                    }

                    if (levelFeat == targetFeature) {
                        bool docGoesRight = (docLeafIndex >> level) & 1;
                        if (leafGoesRight != docGoesRight) {
                            pathProb_with = 0.0f;
                        }
                        float p = leafGoesRight ? rightFrac[level] : (1.0f - rightFrac[level]);
                        pathProb_without *= p;
                    } else if (inS) {
                        bool docGoesRight = (docLeafIndex >> level) & 1;
                        if (leafGoesRight != docGoesRight) {
                            pathProb_with = 0.0f;
                            pathProb_without = 0.0f;
                        }
                    } else {
                        float p = leafGoesRight ? rightFrac[level] : (1.0f - rightFrac[level]);
                        pathProb_with *= p;
                        pathProb_without *= p;
                    }
                }

                for (ui32 k = 0; k < approxDim; ++k) {
                    float leafVal = tree.LeafValues[leaf * approxDim + k];
                    ev_with[k] += pathProb_with * leafVal;
                    ev_without[k] += pathProb_without * leafVal;
                }
            }

            for (ui32 k = 0; k < approxDim; ++k) {
                shapValues[targetFeature][k] += static_cast<float>(weight) * (ev_with[k] - ev_without[k]);
            }
        }
    }
}

// Compute TreeSHAP values for all documents across all trees.
// Returns shapValues[numDocs][numFeatures][approxDim] and expectedValue[approxDim].
static void ComputeAllShapValues(
    const TModel& model,
    const TPackedData& packed,
    const std::vector<uint32_t>& compressedDataFlat,
    ui32 numDocs,
    std::vector<std::vector<std::vector<float>>>& shapValues,  // [numDocs][numFeatures][approxDim]
    std::vector<float>& expectedValue  // [approxDim]
) {
    ui32 numFeatures = model.Features.size();
    ui32 approxDim = model.Info.ApproxDimension;
    shapValues.assign(numDocs,
        std::vector<std::vector<float>>(numFeatures,
            std::vector<float>(approxDim, 0.0f)));
    expectedValue.assign(approxDim, 0.0f);

    // Include base prediction in expected value
    for (ui32 k = 0; k < approxDim && k < model.Info.BasePrediction.size(); ++k) {
        expectedValue[k] += model.Info.BasePrediction[k];
    }

    for (ui32 ti = 0; ti < model.Trees.size(); ++ti) {
        const auto& tree = model.Trees[ti];
        ui32 depth = tree.Depth;
        ui32 numLeaves = 1u << depth;

        // Build the level → feature index mapping
        std::vector<ui32> levelToFeature(depth);
        for (ui32 level = 0; level < depth; ++level) {
            levelToFeature[level] = tree.Splits[level].FeatureIdx;
        }

        // Compute right fractions for this tree
        auto rightFrac = ComputeLevelRightFractions(tree, packed, compressedDataFlat, numDocs);

        // Compute expected value for this tree per dimension
        for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
            float pathProb = 1.0f;
            for (ui32 level = 0; level < depth; ++level) {
                bool goesRight = (leaf >> level) & 1;
                pathProb *= goesRight ? rightFrac[level] : (1.0f - rightFrac[level]);
            }
            for (ui32 k = 0; k < approxDim; ++k) {
                expectedValue[k] += pathProb * tree.LeafValues[leaf * approxDim + k];
            }
        }

        // Compute leaf indices for all docs (CPU version)
        std::vector<ui32> docLeafIndices(numDocs, 0);
        for (ui32 d = 0; d < numDocs; ++d) {
            ui32 leafIdx = 0;
            for (ui32 level = 0; level < depth; ++level) {
                const auto& split = tree.Splits[level];
                ui32 featureIdx = split.FeatureIdx;
                if (featureIdx >= packed.Features.size()) continue;
                const auto& feat = packed.Features[featureIdx];
                ui32 rawMask = feat.Mask >> feat.Shift;
                uint32_t word = compressedDataFlat[d * packed.NumUi32PerDoc + feat.Offset];
                uint32_t binVal = (word >> feat.Shift) & rawMask;
                bool goRight = split.IsOneHot ? (binVal == split.BinThreshold) : (binVal > split.BinThreshold);
                if (goRight) leafIdx |= (1u << level);
            }
            docLeafIndices[d] = leafIdx;
        }

        // Compute SHAP for each doc
        for (ui32 d = 0; d < numDocs; ++d) {
            ComputeTreeShapForDoc(tree, rightFrac, levelToFeature,
                                  docLeafIndices[d], numFeatures, approxDim, shapValues[d]);
        }

        if ((ti + 1) % 50 == 0 || ti + 1 == model.Trees.size()) {
            printf("  SHAP: processed %u/%zu trees\n", ti + 1, model.Trees.size());
        }
    }
}

// ============================================================================
// Main prediction loop
// ============================================================================

int main(int argc, char** argv) {
    auto config = ParseArgs(argc, argv);

    printf("CatBoost-MLX CSV Prediction Tool\n");
    printf("================================\n");

    // Load model
    auto model = LoadModelJSON(config.ModelPath);
    printf("Loaded model: %zu trees, %zu features, loss=%s, approxDim=%u\n",
           model.Trees.size(), model.Features.size(),
           model.Info.LossType.c_str(), model.Info.ApproxDimension);

    // Load CSV data
    auto ds = LoadPredictCSV(config.CsvPath, config.TargetCol, config.GroupCol);
    printf("Loaded data: %u rows, %u features from %s\n", ds.NumDocs, ds.NumFeatures, config.CsvPath.c_str());

    // Apply CTR transformations if the model has CTR features
    if (!model.CtrFeatures.empty()) {
        if (config.Verbose) {
            printf("Applying %zu CTR features\n", model.CtrFeatures.size());
        }

        // Group CTR features by original feature index
        std::unordered_map<ui32, std::vector<size_t>> ctrsByOrigFeature;
        for (size_t ci = 0; ci < model.CtrFeatures.size(); ++ci) {
            ctrsByOrigFeature[model.CtrFeatures[ci].OriginalFeatureIdx].push_back(ci);
        }

        // For each group: first CTR replaces the original feature in-place,
        // extra CTRs are appended as new features
        for (const auto& [origIdx, ctrIndices] : ctrsByOrigFeature) {
            if (origIdx >= ds.NumFeatures) continue;

            // Convert each doc's raw string → CTR float for ALL CTRs in this group at once
            // (before overwriting the raw feature column)
            std::vector<std::vector<std::string>> ctrColumns(ctrIndices.size());
            for (size_t ci = 0; ci < ctrIndices.size(); ++ci) {
                ctrColumns[ci].resize(ds.NumDocs);
            }

            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                const auto& rawVal = ds.RawFeatures[origIdx][d];
                for (size_t ci = 0; ci < ctrIndices.size(); ++ci) {
                    const auto& ctr = model.CtrFeatures[ctrIndices[ci]];
                    auto hashIt = ctr.CatHashMap.find(rawVal);
                    float ctrVal = ctr.DefaultValue;
                    if (hashIt != ctr.CatHashMap.end()) {
                        auto fvIt = ctr.FinalValues.find(hashIt->second);
                        if (fvIt != ctr.FinalValues.end()) {
                            ctrVal = fvIt->second;
                        }
                    }
                    ctrColumns[ci][d] = std::to_string(ctrVal);
                }
            }

            // First CTR replaces the original feature in-place
            ds.RawFeatures[origIdx] = std::move(ctrColumns[0]);

            // Extra CTRs are appended
            for (size_t ci = 1; ci < ctrIndices.size(); ++ci) {
                ds.RawFeatures.push_back(std::move(ctrColumns[ci]));
                ds.FeatureNames.push_back(model.CtrFeatures[ctrIndices[ci]].Name);
                ds.NumFeatures++;
            }
        }
    }

    // Validate feature count (after CTR transformation)
    if (ds.NumFeatures != model.Features.size()) {
        fprintf(stderr, "Error: Data has %u features, model expects %zu features\n",
                ds.NumFeatures, model.Features.size());
        exit(1);
    }

    // Quantize and pack features using model's borders
    auto packed = QuantizeAndPack(ds, model, config.Verbose);

    // Create compressed data MLX array
    auto compressedData = mx::array(
        reinterpret_cast<const int32_t*>(packed.Data.data()),
        {static_cast<int>(ds.NumDocs), static_cast<int>(packed.NumUi32PerDoc)},
        mx::uint32
    );

    // Initialize cursor with base prediction
    ui32 approxDim = model.Info.ApproxDimension;
    ui32 numDocs = ds.NumDocs;
    mx::array cursor = mx::array(0.0f);
    const auto& basePred = model.Info.BasePrediction;
    bool hasBasePred = !basePred.empty();
    if (hasBasePred) {
        for (float v : basePred) { if (std::fabs(v) <= 1e-10f) { /* check at least one non-zero */ } }
    }
    if (hasBasePred && approxDim == 1 && basePred.size() >= 1) {
        cursor = mx::full({static_cast<int>(numDocs)}, basePred[0], mx::float32);
    } else if (hasBasePred && approxDim > 1 && basePred.size() >= approxDim) {
        std::vector<float> initData(approxDim * numDocs);
        for (ui32 k = 0; k < approxDim; ++k)
            for (ui32 d = 0; d < numDocs; ++d)
                initData[k * numDocs + d] = basePred[k];
        cursor = mx::array(initData.data(),
            {static_cast<int>(approxDim), static_cast<int>(numDocs)}, mx::float32);
    } else {
        cursor = (approxDim == 1)
            ? mx::zeros({static_cast<int>(numDocs)}, mx::float32)
            : mx::zeros({static_cast<int>(approxDim), static_cast<int>(numDocs)}, mx::float32);
    }

    // Apply all trees
    for (ui32 ti = 0; ti < model.Trees.size(); ++ti) {
        const auto& tree = model.Trees[ti];
        auto splits = BuildSplitLevels(tree, packed);

        auto leafIndices = ComputeLeafIndices(compressedData, splits, ds.NumDocs);

        ui32 numLeaves = 1u << tree.Depth;
        mx::array leafValues = mx::array(0.0f);
        if (approxDim == 1) {
            leafValues = mx::array(tree.LeafValues.data(),
                {static_cast<int>(numLeaves)}, mx::float32);
        } else {
            leafValues = mx::array(tree.LeafValues.data(),
                {static_cast<int>(numLeaves), static_cast<int>(approxDim)}, mx::float32);
        }

        auto docLeafValues = mx::take(leafValues, mx::astype(leafIndices, mx::int32), 0);
        if (approxDim > 1) {
            docLeafValues = mx::transpose(docLeafValues);  // [K, numDocs]
            cursor = mx::add(cursor, docLeafValues);
        } else {
            cursor = mx::add(cursor, docLeafValues);
        }
    }
    mx::eval(cursor);

    // Transform predictions based on loss type
    const std::string& lossType = model.Info.LossType;

    // Prepare output
    FILE* outFile = stdout;
    bool closeFile = false;
    if (!config.OutputPath.empty()) {
        outFile = fopen(config.OutputPath.c_str(), "w");
        if (!outFile) {
            fprintf(stderr, "Error: Cannot open output file: %s\n", config.OutputPath.c_str());
            exit(1);
        }
        closeFile = true;
    }

    if (lossType == "logloss") {
        // Binary classification: sigmoid → probability
        auto probs = mx::sigmoid(mx::reshape(cursor, {static_cast<int>(numDocs)}));
        mx::eval(probs);
        const float* pPtr = probs.data<float>();

        fprintf(outFile, "probability,predicted_class\n");
        for (ui32 d = 0; d < numDocs; ++d) {
            int predClass = (pPtr[d] >= 0.5f) ? 1 : 0;
            fprintf(outFile, "%.6f,%d\n", pPtr[d], predClass);
        }

        // Evaluation
        if (ds.HasTargets) {
            ui32 correct = 0;
            for (ui32 d = 0; d < numDocs; ++d) {
                int predClass = (pPtr[d] >= 0.5f) ? 1 : 0;
                if (predClass == static_cast<int>(ds.Targets[d])) correct++;
            }
            printf("\nEvaluation: accuracy=%.4f (%u/%u)\n",
                   static_cast<float>(correct) / numDocs, correct, numDocs);
        }
    } else if (lossType == "multiclass") {
        // Multiclass: softmax → class probabilities
        auto sm = ComputeSoftmax(cursor, approxDim, numDocs);
        mx::eval(sm.Probs);
        const float* probsPtr = sm.Probs.data<float>();

        // Also compute implicit class 0 probability
        // probs from softmax are for classes 1..K-1, class 0 = 1 - sum
        ui32 numClasses = model.Info.NumClasses;

        // Header
        fprintf(outFile, "predicted_class");
        for (ui32 k = 0; k < numClasses; ++k) fprintf(outFile, ",prob_class_%u", k);
        fprintf(outFile, "\n");

        // Compute full probabilities including implicit class (class K-1)
        // sm.Probs is [approxDim, numDocs] = [K-1, numDocs]
        // cursor dim k → class k (k=0..K-2), implicit class (logit=0) → class K-1
        ui32 correct = 0;
        for (ui32 d = 0; d < numDocs; ++d) {
            float sumProb = 0.0f;
            std::vector<float> classProbs(numClasses);
            for (ui32 k = 0; k < approxDim; ++k) {
                classProbs[k] = probsPtr[k * numDocs + d];
                sumProb += classProbs[k];
            }
            classProbs[numClasses - 1] = std::max(0.0f, 1.0f - sumProb);  // implicit class = last class

            // Predicted class = argmax
            ui32 predClass = 0;
            float maxProb = classProbs[0];
            for (ui32 k = 1; k < numClasses; ++k) {
                if (classProbs[k] > maxProb) { maxProb = classProbs[k]; predClass = k; }
            }

            fprintf(outFile, "%u", predClass);
            for (ui32 k = 0; k < numClasses; ++k) fprintf(outFile, ",%.6f", classProbs[k]);
            fprintf(outFile, "\n");

            if (ds.HasTargets && predClass == static_cast<ui32>(ds.Targets[d])) correct++;
        }

        if (ds.HasTargets) {
            printf("\nEvaluation: accuracy=%.4f (%u/%u)\n",
                   static_cast<float>(correct) / numDocs, correct, numDocs);
        }
    } else if (lossType == "poisson" || lossType == "tweedie") {
        // Count/continuous regression with log link: apply exp() to get expected value
        auto flatCursor = mx::reshape(cursor, {static_cast<int>(numDocs)});
        auto predictions = mx::exp(flatCursor);
        mx::eval(predictions);
        const float* preds = predictions.data<float>();

        fprintf(outFile, "prediction\n");
        for (ui32 d = 0; d < numDocs; ++d) {
            fprintf(outFile, "%.6f\n", preds[d]);
        }

        if (ds.HasTargets) {
            float sumSqErr = 0.0f, sumAbsErr = 0.0f;
            for (ui32 d = 0; d < numDocs; ++d) {
                float err = preds[d] - ds.Targets[d];
                sumSqErr += err * err;
                sumAbsErr += std::fabs(err);
            }
            float rmse = std::sqrt(sumSqErr / numDocs);
            float mae = sumAbsErr / numDocs;
            printf("\nEvaluation: RMSE=%.6f, MAE=%.6f\n", rmse, mae);
        }
    } else {
        // Regression: raw values (rmse, mae, quantile, huber, mape)
        auto flatCursor = mx::reshape(cursor, {static_cast<int>(numDocs)});
        mx::eval(flatCursor);
        const float* preds = flatCursor.data<float>();

        fprintf(outFile, "prediction\n");
        for (ui32 d = 0; d < numDocs; ++d) {
            fprintf(outFile, "%.6f\n", preds[d]);
        }

        // Evaluation
        if (ds.HasTargets) {
            float sumSqErr = 0.0f, sumAbsErr = 0.0f;
            for (ui32 d = 0; d < numDocs; ++d) {
                float err = preds[d] - ds.Targets[d];
                sumSqErr += err * err;
                sumAbsErr += std::fabs(err);
            }
            float rmse = std::sqrt(sumSqErr / numDocs);
            float mae = sumAbsErr / numDocs;
            printf("\nEvaluation: RMSE=%.6f, MAE=%.6f\n", rmse, mae);
        }
    }

    // ── SHAP values ─────────────────────────────────────────────────────
    if (config.ComputeShap) {
        printf("\nComputing SHAP values...\n");
        std::vector<std::vector<std::vector<float>>> shapValues;
        std::vector<float> expectedValue;
        ComputeAllShapValues(model, packed, packed.Data, numDocs,
                             shapValues, expectedValue);

        // Determine SHAP output file
        std::string shapPath;
        if (!config.OutputPath.empty()) {
            auto dotPos = config.OutputPath.rfind('.');
            if (dotPos != std::string::npos) {
                shapPath = config.OutputPath.substr(0, dotPos) + "_shap" + config.OutputPath.substr(dotPos);
            } else {
                shapPath = config.OutputPath + "_shap";
            }
        }

        FILE* shapFile = stdout;
        bool closeShapFile = false;
        if (!shapPath.empty()) {
            shapFile = fopen(shapPath.c_str(), "w");
            if (!shapFile) {
                fprintf(stderr, "Error: Cannot open SHAP output file: %s\n", shapPath.c_str());
            } else {
                closeShapFile = true;
            }
        }

        if (shapFile) {
            // Header
            if (approxDim == 1) {
                for (ui32 f = 0; f < model.Features.size(); ++f) {
                    if (f > 0) fprintf(shapFile, ",");
                    fprintf(shapFile, "%s_shap", model.Features[f].Name.c_str());
                }
                fprintf(shapFile, ",expected_value,prediction\n");
            } else {
                bool first = true;
                for (ui32 f = 0; f < model.Features.size(); ++f) {
                    for (ui32 k = 0; k < approxDim; ++k) {
                        if (!first) fprintf(shapFile, ",");
                        fprintf(shapFile, "%s_shap_class%u", model.Features[f].Name.c_str(), k);
                        first = false;
                    }
                }
                for (ui32 k = 0; k < approxDim; ++k)
                    fprintf(shapFile, ",expected_value_class%u", k);
                for (ui32 k = 0; k < approxDim; ++k)
                    fprintf(shapFile, ",prediction_class%u", k);
                fprintf(shapFile, "\n");
            }

            // Get raw predictions (before any transform like sigmoid/softmax)
            mx::eval(cursor);
            const float* cursorPtr = cursor.data<float>();

            // Data rows
            for (ui32 d = 0; d < numDocs; ++d) {
                if (approxDim == 1) {
                    for (ui32 f = 0; f < model.Features.size(); ++f) {
                        if (f > 0) fprintf(shapFile, ",");
                        fprintf(shapFile, "%.6f", shapValues[d][f][0]);
                    }
                    fprintf(shapFile, ",%.6f,%.6f\n", expectedValue[0], cursorPtr[d]);
                } else {
                    bool first = true;
                    for (ui32 f = 0; f < model.Features.size(); ++f) {
                        for (ui32 k = 0; k < approxDim; ++k) {
                            if (!first) fprintf(shapFile, ",");
                            fprintf(shapFile, "%.6f", shapValues[d][f][k]);
                            first = false;
                        }
                    }
                    for (ui32 k = 0; k < approxDim; ++k)
                        fprintf(shapFile, ",%.6f", expectedValue[k]);
                    for (ui32 k = 0; k < approxDim; ++k)
                        fprintf(shapFile, ",%.6f", cursorPtr[k * numDocs + d]);
                    fprintf(shapFile, "\n");
                }
            }

            if (closeShapFile) {
                fclose(shapFile);
                printf("SHAP values written to: %s\n", shapPath.c_str());
            }

            // Verify sum property per dimension
            float maxErr = 0.0f;
            for (ui32 d = 0; d < numDocs; ++d) {
                for (ui32 k = 0; k < approxDim; ++k) {
                    float shapSum = 0.0f;
                    for (ui32 f = 0; f < model.Features.size(); ++f) {
                        shapSum += shapValues[d][f][k];
                    }
                    float pred = (approxDim == 1) ? cursorPtr[d] : cursorPtr[k * numDocs + d];
                    float err = std::fabs(shapSum + expectedValue[k] - pred);
                    if (err > maxErr) maxErr = err;
                }
            }
            printf("SHAP sum property check: max |sum(shap) + E[f(x)] - f(x)| = %.2e\n", maxErr);
        }
    }

    if (closeFile) {
        fclose(outFile);
        printf("Predictions written to: %s\n", config.OutputPath.c_str());
    }

    return 0;
}
