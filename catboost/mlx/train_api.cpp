// catboost/mlx/train_api.cpp — TrainFromArrays implementation.
//
// This file provides the in-memory training entry point for nanobind bindings.
// It includes csv_train.cpp (with main() excluded) to reuse all internal
// functions and types, then wraps them behind the public TTrainConfig /
// TTrainResultAPI API defined in train_api.h.
//
// Build:
//   - For nanobind library: compile this file (NOT csv_train.cpp) as the
//     single translation unit.  All csv_train functions are included here.
//   - For CLI binary: compile csv_train.cpp directly (without CATBOOST_MLX_NO_MAIN).

#define CATBOOST_MLX_NO_MAIN
#include "catboost/mlx/tests/csv_train.cpp"

#include "catboost/mlx/train_api.h"

#include <stdexcept>   // std::runtime_error — BUG-007 groupIds sortedness contract

// ============================================================================
// TTrainConfig → TConfig conversion
// ============================================================================

static TConfig TrainConfigToInternal(const TTrainConfig& tc) {
    // S28-LG-GUARD (C++ defense-in-depth): Cosine+Lossguide combination rejected until S29 RCA.
    // Mirrors python/catboost_mlx/core.py:628-636 verbatim so TODO markers stay greppable
    // across languages. nanobind auto-translates std::invalid_argument → Python ValueError.
    if (tc.ScoreFunction == "Cosine" && tc.GrowPolicy == "Lossguide") {
        throw std::invalid_argument(
            "score_function='Cosine' with grow_policy='Lossguide' is not supported "
            "in catboost-mlx: priority-queue leaf ordering interacts with Cosine "
            "joint-gain magnitude producing unacceptable per-partition gain drift "
            "vs CPU CatBoost. Root-cause investigation is scheduled for Sprint 29 "
            "(TODO-S29-LG-COSINE-RCA). Use score_function='L2' with Lossguide, or "
            "switch grow_policy to 'SymmetricTree' or 'Depthwise' for Cosine."
        );
    }
    TConfig c;
    c.NumIterations        = tc.NumIterations;
    c.MaxDepth             = tc.MaxDepth;
    c.LearningRate         = tc.LearningRate;
    c.L2RegLambda          = tc.L2RegLambda;
    c.MaxBins              = tc.MaxBins;
    c.LossType             = tc.LossType;
    c.GroupCol             = tc.GroupCol;
    c.WeightCol            = tc.WeightCol;
    c.CatFeatureCols       = tc.CatFeatureCols;
    c.EvalFraction         = tc.EvalFraction;
    c.EarlyStoppingPatience = tc.EarlyStoppingPatience;
    c.SubsampleRatio       = tc.SubsampleRatio;
    c.ColsampleByTree      = tc.ColsampleByTree;
    c.RandomSeed           = tc.RandomSeed;
    c.RandomStrength       = tc.RandomStrength;
    c.BootstrapType        = tc.BootstrapType;
    c.BaggingTemperature   = tc.BaggingTemperature;
    c.MvsReg               = tc.MvsReg;
    c.NanMode              = tc.NanMode;
    c.UseCtr               = tc.UseCtr;
    c.CtrPrior             = tc.CtrPrior;
    c.MaxOneHotSize        = tc.MaxOneHotSize;
    c.MinDataInLeaf        = tc.MinDataInLeaf;
    c.MonotoneConstraints  = tc.MonotoneConstraints;
    c.GrowPolicy           = tc.GrowPolicy;
    c.MaxLeaves            = tc.MaxLeaves;
    c.ScoreFunction        = tc.ScoreFunction;
    c.SnapshotPath         = tc.SnapshotPath;
    c.SnapshotInterval     = tc.SnapshotInterval;
    c.Verbose              = tc.Verbose;
    c.ShowFeatureImportance = tc.ComputeFeatureImportance;
    // CLI-specific fields left at defaults:
    // c.CsvPath, c.TargetCol, c.OutputModelPath, c.CVFolds, c.EvalFile
    return c;
}

// ============================================================================
// Build TDataset from flat arrays
// ============================================================================

static TDataset BuildDatasetFromArrays(
    const float* features,       // row-major [numDocs * numFeatures]
    const float* targets,        // [numDocs]
    const std::vector<std::string>& featureNames,
    const std::vector<bool>& isCategorical,
    const std::vector<float>& weights,
    const std::vector<uint32_t>& groupIds,
    const std::vector<std::unordered_map<std::string, uint32_t>>& catHashMaps,
    uint32_t numDocs,
    uint32_t numFeatures
) {
    TDataset ds;
    ds.NumDocs = numDocs;
    ds.NumFeatures = numFeatures;

    // Feature names (use defaults if empty)
    ds.FeatureNames.resize(numFeatures);
    for (uint32_t f = 0; f < numFeatures; ++f) {
        ds.FeatureNames[f] = (f < featureNames.size() && !featureNames[f].empty())
            ? featureNames[f] : ("f" + std::to_string(f));
    }

    // Categorical flags
    ds.IsCategorical.resize(numFeatures, false);
    for (uint32_t f = 0; f < numFeatures && f < isCategorical.size(); ++f) {
        ds.IsCategorical[f] = isCategorical[f];
    }

    // Cat hash maps
    ds.CatHashMaps.resize(numFeatures);
    for (uint32_t f = 0; f < numFeatures && f < catHashMaps.size(); ++f) {
        ds.CatHashMaps[f] = catHashMaps[f];
    }

    // Features: convert row-major [numDocs, numFeatures] → column-major [numFeatures][numDocs]
    ds.Features.resize(numFeatures);
    ds.HasNaN.resize(numFeatures, false);
    for (uint32_t f = 0; f < numFeatures; ++f) {
        ds.Features[f].resize(numDocs);
        for (uint32_t d = 0; d < numDocs; ++d) {
            float val = features[d * numFeatures + f];
            ds.Features[f][d] = val;
            if (std::isnan(val)) ds.HasNaN[f] = true;
        }
    }

    // Targets
    ds.Targets.assign(targets, targets + numDocs);

    // Weights
    if (!weights.empty()) {
        ds.Weights = weights;
    }

    // Group IDs → group offsets.
    // BUG-007 contract: groupIds must be sorted non-decreasing. GroupOffsets is
    // built via consecutive-equal detection below, so shuffled input would produce
    // wrong offsets and silently diverge from the subprocess (csv_train) path.
    // The Python wrapper (core.py:_fit_nanobind) sorts before calling; this check
    // fails loudly for any direct C++ caller that bypasses it. See KNOWN_BUGS.md.
    if (!groupIds.empty()) {
        for (uint32_t d = 1; d < numDocs; ++d) {
            if (groupIds[d] < groupIds[d - 1]) {
                throw std::runtime_error(
                    "BuildDatasetFromArrays: groupIds must be sorted non-decreasing "
                    "(BUG-007 contract). GroupOffsets is built from consecutive-equal "
                    "detection; unsorted input produces wrong offsets and silent "
                    "ranking-loss divergence. Sort groupIds (and apply the same "
                    "permutation to features/targets/weights) before calling."
                );
            }
        }
        ds.GroupIds.resize(numDocs);
        for (uint32_t d = 0; d < numDocs; ++d) {
            ds.GroupIds[d] = groupIds[d];
        }
        ds.GroupOffsets.push_back(0);
        for (uint32_t d = 1; d < numDocs; ++d) {
            if (groupIds[d] != groupIds[d - 1]) {
                ds.GroupOffsets.push_back(d);
                ds.NumGroups++;
            }
        }
        ds.GroupOffsets.push_back(numDocs);
        ds.NumGroups++;
    }

    return ds;
}

// ============================================================================
// Feature importance calculation
// ============================================================================

static std::unordered_map<std::string, double> CalcFeatureImportanceMap(
    const std::vector<TTreeRecord>& trees,
    const std::vector<std::string>& featureNames,
    uint32_t numFeatures
) {
    std::vector<double> featureGain(numFeatures, 0.0);
    for (const auto& tree : trees) {
        for (uint32_t level = 0; level < tree.SplitProps.size(); ++level) {
            uint32_t featIdx = tree.SplitProps[level].FeatureId;
            if (featIdx < numFeatures) {
                featureGain[featIdx] += tree.SplitProps[level].Gain;
            }
        }
    }

    std::unordered_map<std::string, double> result;
    for (uint32_t f = 0; f < numFeatures; ++f) {
        if (featureGain[f] > 0.0) {
            std::string name = (f < featureNames.size() && !featureNames[f].empty())
                ? featureNames[f] : ("f" + std::to_string(f));
            result[name] = featureGain[f];
        }
    }
    return result;
}

// ============================================================================
// TrainFromArrays — public entry point
// ============================================================================

TTrainResultAPI TrainFromArrays(
    const float*        features,
    const float*        targets,
    const std::vector<std::string>&   featureNames,
    const std::vector<bool>&          isCategorical,
    const std::vector<float>&         weights,
    const std::vector<uint32_t>&      groupIds,
    const std::vector<std::unordered_map<std::string, uint32_t>>& catHashMaps,
    uint32_t            numDocs,
    uint32_t            numFeatures,
    const float*        valFeatures,
    const float*        valTargets,
    uint32_t            valDocs,
    const TTrainConfig& apiConfig
) {
    // Convert public config → internal config
    TConfig config = TrainConfigToInternal(apiConfig);

    // Build dataset from flat arrays
    TDataset ds = BuildDatasetFromArrays(
        features, targets, featureNames, isCategorical,
        weights, groupIds, catHashMaps, numDocs, numFeatures);

    // Parse loss type (may include parameter like "quantile:0.75")
    auto lossConfig = ParseLossType(config.LossType);
    std::string lossType = lossConfig.Type;
    float lossParam = lossConfig.Param;

    if (lossType == "auto") {
        lossType = DetectLossType(ds.Targets);
    }

    bool isRankingLoss = (lossType == "pairlogit" || lossType == "yetirank");

    // Determine approx dimension
    uint32_t approxDim = 1;
    uint32_t numClasses = 0;
    if (lossType == "multiclass") {
        float maxTarget = *std::max_element(ds.Targets.begin(), ds.Targets.end());
        numClasses = static_cast<uint32_t>(maxTarget) + 1;
        approxDim = numClasses - 1;
    }

    // CTR target encoding for high-cardinality categoricals
    std::vector<TCtrFeature> ctrFeatures;
    if (config.UseCtr) {
        ctrFeatures = ComputeCtrFeatures(ds, lossType, numClasses,
                                          config.CtrPrior, config.MaxOneHotSize, config.RandomSeed);
    }

    // Quantize and pack features
    auto quant = QuantizeFeatures(ds, config.MaxBins);
    auto packed = PackFeatures(quant, ds);

#ifdef CATBOOST_MLX_DEBUG_LEAF
    // P10 — quantization borders for features 0 and 1 (signal-bearing columns)
    // Prints border count + min/max + first/last 5 borders so we can diff against
    // CatBoost CPU's GreedyLogSum borders to confirm or falsify H2 (border divergence).
    for (ui32 p10_f = 0; p10_f < std::min(static_cast<ui32>(quant.Borders.size()), 2u); ++p10_f) {
        const auto& b = quant.Borders[p10_f];
        printf("[DBG P10] feature_%u borders: count=%zu", p10_f, b.size());
        if (!b.empty()) {
            printf("  min=%.6f  max=%.6f", b.front(), b.back());
            printf("\n[DBG P10] feature_%u first5:", p10_f);
            for (ui32 i = 0; i < std::min((ui32)b.size(), 5u); ++i) printf(" %.6f", b[i]);
            printf("\n[DBG P10] feature_%u last5: ", p10_f);
            ui32 start = b.size() >= 5 ? (ui32)b.size() - 5 : 0;
            for (ui32 i = start; i < (ui32)b.size(); ++i) printf(" %.6f", b[i]);
        }
        printf("\n");
    }
    fflush(stdout);
#endif

    if (packed.TotalBinFeatures == 0) {
        TTrainResultAPI result;
        result.ModelJSON = "{}";
        return result;
    }

    // --- Build validation dataset if explicit val arrays provided ---
    TDataset valDs;
    TQuantization valQuant;
    TPackedData valPacked;
    uint32_t actualValDocs = 0;

    if (valFeatures && valTargets && valDocs > 0) {
        // Build val dataset from arrays
        std::vector<bool> valIsCat = isCategorical;
        std::vector<std::unordered_map<std::string, uint32_t>> valCatMaps = catHashMaps;
        valDs = BuildDatasetFromArrays(
            valFeatures, valTargets, featureNames, valIsCat,
            {}, {}, valCatMaps, valDocs, numFeatures);

        // Apply CTR to val data using training stats
        if (!ctrFeatures.empty()) {
            ApplyCtrToEvalData(valDs, ctrFeatures, ds);
        }

        valQuant = QuantizeWithBorders(valDs, quant, ds);
        valPacked = PackFeatures(valQuant, valDs);
        actualValDocs = valDocs;
    }

    // --- Handle eval-fraction split (train/val from single dataset) ---
    uint32_t trainDocs = numDocs;
    uint32_t splitValDocs = 0;

    // Ranking data for RunTraining
    std::vector<float> trainTargetsVec, valTargetsVec;
    std::vector<uint32_t> trainGroupOffsets, valGroupOffsets;
    uint32_t trainNumGroups = 0, valNumGroups = 0;

    if (actualValDocs == 0 && config.EvalFraction > 0.0f && config.EvalFraction < 1.0f) {
        if (isRankingLoss && ds.NumGroups > 0) {
            uint32_t valGroups = static_cast<uint32_t>(ds.NumGroups * config.EvalFraction);
            if (valGroups == 0) valGroups = 1;
            uint32_t trainGroups = ds.NumGroups - valGroups;
            if (trainGroups == 0) { trainGroups = 1; valGroups = ds.NumGroups - 1; }
            trainDocs = ds.GroupOffsets[trainGroups];
            splitValDocs = numDocs - trainDocs;
            trainNumGroups = trainGroups;
            valNumGroups = valGroups;
            trainGroupOffsets.assign(ds.GroupOffsets.begin(), ds.GroupOffsets.begin() + trainGroups + 1);
            for (uint32_t g = 0; g <= valGroups; ++g) {
                valGroupOffsets.push_back(ds.GroupOffsets[trainGroups + g] - trainDocs);
            }
        } else {
            splitValDocs = static_cast<uint32_t>(numDocs * config.EvalFraction);
            trainDocs = numDocs - splitValDocs;
        }
    }

    if (isRankingLoss) {
        if (actualValDocs > 0) {
            trainTargetsVec.assign(ds.Targets.begin(), ds.Targets.end());
            trainGroupOffsets = ds.GroupOffsets;
            trainNumGroups = ds.NumGroups;
            valTargetsVec.assign(valDs.Targets.begin(), valDs.Targets.end());
            valGroupOffsets = valDs.GroupOffsets;
            valNumGroups = valDs.NumGroups;
        } else {
            trainTargetsVec.assign(ds.Targets.begin(), ds.Targets.begin() + trainDocs);
            if (splitValDocs > 0) {
                valTargetsVec.assign(ds.Targets.begin() + trainDocs, ds.Targets.end());
            }
            if (splitValDocs == 0) {
                trainGroupOffsets = ds.GroupOffsets;
                trainNumGroups = ds.NumGroups;
            }
        }
    }

    // --- Transfer to GPU ---
    mx::array compressedData = mx::array(0);
    mx::array targetsArr = mx::array(0.0f);
    mx::array valCompressedData = mx::array(0);
    mx::array valTargetsArr = mx::array(0.0f);

    if (actualValDocs > 0) {
        // Explicit validation set
        compressedData = mx::array(
            reinterpret_cast<const int32_t*>(packed.Data.data()),
            {static_cast<int>(trainDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        targetsArr = mx::array(ds.Targets.data(), {static_cast<int>(trainDocs)}, mx::float32);

        valCompressedData = mx::array(
            reinterpret_cast<const int32_t*>(valPacked.Data.data()),
            {static_cast<int>(actualValDocs), static_cast<int>(valPacked.NumUi32PerDoc)}, mx::uint32);
        valTargetsArr = mx::array(valDs.Targets.data(), {static_cast<int>(actualValDocs)}, mx::float32);
        mx::eval({compressedData, targetsArr, valCompressedData, valTargetsArr});
    } else if (splitValDocs > 0) {
        // Eval-fraction split
        std::vector<uint32_t> trainData(trainDocs * packed.NumUi32PerDoc);
        std::vector<uint32_t> valData(splitValDocs * packed.NumUi32PerDoc);
        std::vector<float> trainTgts(trainDocs);
        std::vector<float> valTgts(splitValDocs);

        for (uint32_t d = 0; d < trainDocs; ++d) {
            for (uint32_t w = 0; w < packed.NumUi32PerDoc; ++w)
                trainData[d * packed.NumUi32PerDoc + w] = packed.Data[d * packed.NumUi32PerDoc + w];
            trainTgts[d] = ds.Targets[d];
        }
        for (uint32_t d = 0; d < splitValDocs; ++d) {
            for (uint32_t w = 0; w < packed.NumUi32PerDoc; ++w)
                valData[d * packed.NumUi32PerDoc + w] = packed.Data[(trainDocs + d) * packed.NumUi32PerDoc + w];
            valTgts[d] = ds.Targets[trainDocs + d];
        }

        compressedData = mx::array(
            reinterpret_cast<const int32_t*>(trainData.data()),
            {static_cast<int>(trainDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        targetsArr = mx::array(trainTgts.data(), {static_cast<int>(trainDocs)}, mx::float32);

        valCompressedData = mx::array(
            reinterpret_cast<const int32_t*>(valData.data()),
            {static_cast<int>(splitValDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        valTargetsArr = mx::array(valTgts.data(), {static_cast<int>(splitValDocs)}, mx::float32);
        mx::eval({compressedData, targetsArr, valCompressedData, valTargetsArr});
    } else {
        // No validation
        compressedData = mx::array(
            reinterpret_cast<const int32_t*>(packed.Data.data()),
            {static_cast<int>(numDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        targetsArr = mx::array(ds.Targets.data(), {static_cast<int>(numDocs)}, mx::float32);
        mx::eval({compressedData, targetsArr});
    }

    // Slice sample weights for training subset
    std::vector<float> trainWeights;
    if (!ds.Weights.empty()) {
        trainWeights.assign(ds.Weights.begin(), ds.Weights.begin() + trainDocs);
    }

    uint32_t effectiveValDocs = (actualValDocs > 0) ? actualValDocs : splitValDocs;

    // --- Run training ---
    auto trainResult = RunTraining(
        config, compressedData, trainDocs, targetsArr,
        valCompressedData, effectiveValDocs, valTargetsArr,
        packed, lossType, lossParam, approxDim, numClasses,
        apiConfig.Verbose,
        trainTargetsVec, trainGroupOffsets, trainNumGroups,
        valTargetsVec, valGroupOffsets, valNumGroups,
        trainWeights,
        config.SnapshotPath, config.SnapshotInterval
    );

    // Truncate trees and loss history to best iteration when early stopping fired
    if (config.EarlyStoppingPatience > 0 && trainResult.BestIteration > 0 &&
        trainResult.BestIteration + 1 < trainResult.Trees.size()) {
        uint32_t keepTrees = trainResult.BestIteration + 1;
        trainResult.Trees.resize(keepTrees);
        trainResult.TreesBuilt = keepTrees;
        // Truncate loss histories to match the saved model length
        if (trainResult.TrainLossHistory.size() > keepTrees)
            trainResult.TrainLossHistory.resize(keepTrees);
        if (trainResult.EvalLossHistory.size() > keepTrees)
            trainResult.EvalLossHistory.resize(keepTrees);
    }

    // --- Package into public result ---
    TTrainResultAPI result;
    result.FinalTrainLoss  = trainResult.FinalTrainLoss;
    result.FinalTestLoss   = trainResult.FinalTestLoss;
    result.BestIteration   = trainResult.BestIteration;
    result.TreesBuilt      = trainResult.TreesBuilt;
    result.GradMs          = trainResult.GradMs;
    result.TreeSearchMs    = trainResult.TreeSearchMs;
    result.LeafMs          = trainResult.LeafMs;
    result.ApplyMs         = trainResult.ApplyMs;

    // Feature names
    result.FeatureNames = ds.FeatureNames;

    // Feature importance
    if (apiConfig.ComputeFeatureImportance && !trainResult.Trees.empty()) {
        result.FeatureImportance = CalcFeatureImportanceMap(
            trainResult.Trees, ds.FeatureNames, ds.NumFeatures);
    }

    // Model JSON
    if (!trainResult.Trees.empty()) {
        result.ModelJSON = BuildModelJSONString(
            trainResult.Trees, ds, quant, lossType, lossParam,
            config, approxDim, numClasses, ctrFeatures, trainResult.BasePrediction);
    } else {
        result.ModelJSON = "{}";
    }

    // Loss history per iteration
    result.TrainLossHistory = trainResult.TrainLossHistory;
    result.EvalLossHistory  = trainResult.EvalLossHistory;

    return result;
}
