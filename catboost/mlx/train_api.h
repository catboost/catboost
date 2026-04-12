// catboost/mlx/train_api.h — Public API for CatBoost-MLX in-memory training.
//
// This header exposes the reusable training entry point (TrainFromArrays)
// for use by nanobind Python bindings. It defines only the PUBLIC types
// needed by callers; internal types (TConfig, TDataset, etc.) live in
// csv_train.cpp and are accessed by train_api.cpp via the include trick.
//
// Design principles:
//   - No MLX or Metal headers — callers (e.g. bindings.cpp) compile without
//     MLX on the include path if they only need the API types.
//   - TTrainConfig mirrors TConfig exactly, minus CLI-specific fields
//     (CsvPath, TargetCol, OutputModelPath, ShowFeatureImportance, CVFolds,
//     EvalFile). Defaults must match TConfig defaults exactly.
//   - TTrainResultAPI is the public result type; the internal TTrainResult
//     (used by RunTraining) stays in csv_train.cpp.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// ============================================================================
// Training hyperparameters — mirrors TConfig minus CLI-specific fields.
// All defaults must match TConfig's defaults exactly.
// ============================================================================

struct TTrainConfig {
    // Core training parameters
    uint32_t NumIterations       = 100;
    uint32_t MaxDepth            = 6;
    float LearningRate           = 0.1f;
    float L2RegLambda            = 3.0f;
    uint32_t MaxBins             = 255;
    std::string LossType         = "auto";

    // Column roles
    // GroupCol indexes into the FEATURE matrix (not counting target). -1 = not present.
    // WeightCol is unused by TrainFromArrays (weights passed as separate array).
    int GroupCol                 = -1;
    int WeightCol                = -1;

    // Categorical features (0-based feature indices)
    std::unordered_set<int> CatFeatureCols;

    // Validation / early stopping
    float EvalFraction           = 0.0f;
    uint32_t EarlyStoppingPatience = 0;

    // Subsampling
    float SubsampleRatio         = 1.0f;
    float ColsampleByTree        = 1.0f;
    uint32_t RandomSeed          = 42;
    float RandomStrength         = 1.0f;

    // Bootstrap
    std::string BootstrapType    = "no";    // "no", "bayesian", "bernoulli", "mvs"
    float BaggingTemperature     = 1.0f;
    float MvsReg                 = 0.0f;

    // NaN handling
    std::string NanMode          = "min";   // "min" or "forbidden"

    // CTR target encoding
    bool UseCtr                  = false;
    float CtrPrior               = 0.5f;
    uint32_t MaxOneHotSize       = 10;

    // Regularization
    uint32_t MinDataInLeaf       = 1;
    std::vector<int> MonotoneConstraints;   // per-feature: 0=none, 1=inc, -1=dec

    // Grow policy
    std::string GrowPolicy       = "SymmetricTree";  // "SymmetricTree", "Depthwise", "Lossguide"
    uint32_t MaxLeaves           = 31;

    // Snapshot save/resume
    std::string SnapshotPath;
    uint32_t SnapshotInterval    = 1;

    // Output control
    bool Verbose                 = false;
    bool ComputeFeatureImportance = false;
};

// ============================================================================
// Training result — public API type returned by TrainFromArrays().
// ============================================================================

struct TTrainResultAPI {
    // Core results
    float FinalTrainLoss         = 0.0f;
    float FinalTestLoss          = 0.0f;
    uint32_t BestIteration       = 0;
    uint32_t TreesBuilt          = 0;

    // Phase timing (ms, accumulated across all iterations)
    double GradMs                = 0;
    double TreeSearchMs          = 0;
    double LeafMs                = 0;
    double ApplyMs               = 0;

    // Model output
    std::string ModelJSON;                                      // complete catboost-mlx-json string
    std::vector<std::string> FeatureNames;                      // echoed back for Python
    std::unordered_map<std::string, double> FeatureImportance;  // name -> total gain
    std::vector<float> TrainLossHistory;                        // per-iteration train loss
    std::vector<float> EvalLossHistory;                         // per-iteration eval loss (empty if no val)
};

// ============================================================================
// Public API
// ============================================================================

/// Train a GBDT model from in-memory arrays (numpy-friendly flat layout).
///
/// @param features       Row-major float32 matrix: [numDocs * numFeatures].
///                       Categorical features are pre-encoded as float(hash_index).
/// @param targets        Float32 vector: [numDocs].
/// @param featureNames   Names for each feature column. Length == numFeatures.
///                       Pass empty vector for "f0", "f1", ... defaults.
/// @param isCategorical  Bool mask: [numFeatures]. true for categorical columns.
/// @param weights        Optional per-sample weights: [numDocs]. Empty = uniform.
/// @param groupIds       Optional group IDs for ranking: [numDocs]. Empty = no groups.
/// @param catHashMaps    Per-feature string->bin maps for categorical features.
///                       Length == numFeatures; empty map for numeric features.
/// @param numDocs        Number of training documents.
/// @param numFeatures    Number of feature columns.
/// @param valFeatures    Optional validation features (row-major): [valDocs * numFeatures].
///                       nullptr to derive val set from EvalFraction in config.
/// @param valTargets     Optional validation targets: [valDocs]. nullptr = no explicit val.
/// @param valDocs        Number of validation documents (0 if no explicit val set).
/// @param config         All training hyperparameters.
///
/// @returns TTrainResultAPI with ModelJSON, FeatureImportance, and loss histories.
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
    const TTrainConfig& config
);
