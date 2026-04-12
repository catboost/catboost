# Nanobind Implementation Specification — Sprint 11

**Status:** Draft  
**Date:** 2026-04-11  
**Authors:** Technical Writer (based on reading csv_train.cpp, core.py, and MLX python/src/)  
**Target agents:** ml-engineer (Phases 1, 2, 4), devops-engineer (Phase 3), qa-engineer (Phase 5)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Phase 1 — Extract csv_train into a Library](#2-phase-1--extract-csv_train-into-a-library-ml-engineer)
3. [Phase 2 — Nanobind Module](#3-phase-2--nanobind-module-ml-engineer)
4. [Phase 3 — Build System](#4-phase-3--build-system-devops-engineer)
5. [Phase 4 — Python Integration](#5-phase-4--python-integration-ml-engineer)
6. [Phase 5 — Testing](#6-phase-5--testing-qa-engineer)
7. [File Inventory](#7-file-inventory)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [Risk Register](#9-risk-register)

---

## 1. Overview

### Goal

Replace the subprocess bridge in `python/catboost_mlx/core.py` with in-process nanobind bindings. Training and prediction will call C++ functions directly from Python, eliminating:

- Temp-file round-trips (CSV/CBMX write → disk → C++ read)
- Process spawn overhead (~50–200 ms per call)
- The PTY complexity needed to stream printf output
- Subprocess security concerns (PATH injection, TOCTOU on temp files)

The fallback to the subprocess path is preserved for environments where the extension cannot be compiled.

### Architecture: Current vs Proposed

**Current (subprocess bridge)**

```
Python fit() call
    │
    ├─ write X, y to /tmp/catboost_mlx_*/train.cbmx  (binary) or train.csv
    │
    ├─ subprocess.Popen(["csv_train", "train.cbmx", "--output", "model.json", ...])
    │       └─ reads file from disk
    │       └─ trains on Metal GPU
    │       └─ writes model.json to disk
    │       └─ writes feature importance to stdout
    │
    ├─ parse stdout (regex) for loss history and feature importance
    ├─ open("model.json") and json.load()
    └─ shutil.rmtree(tmpdir)
```

**Proposed (nanobind in-process)**

```
Python fit() call
    │
    ├─ validate inputs (unchanged)
    │
    ├─ import catboost_mlx._core  (nanobind .so)
    │
    ├─ _core.train(
    │       features=X_np,          # numpy float32 array, zero-copy when contiguous
    │       targets=y_np,           # numpy float32 array
    │       feature_names=[...],
    │       cat_feature_indices=[...],
    │       weights=sw_np,          # optional
    │       group_ids=gid_np,       # optional
    │       config=TTrainConfig{...}
    │   )  → TTrainResult { model_json: str, feature_importance: dict, ... }
    │
    └─ json.loads(result.model_json)  → self._model_data
```

### Key design principle

Wrap the internals of `catboost/mlx/tests/csv_train.cpp` — **not** `catboost/mlx/methods/mlx_boosting.h`. The csv_train.cpp codepath (4083 lines) is the battle-tested implementation that handles quantization, packing, CTR encoding, cross-validation, snapshot resume, ranking losses, and all three grow policies. `mlx_boosting.h` is a separate library path with different data structures; mixing the two would require significant integration work and create two models to maintain.

---

## 2. Phase 1 — Extract csv_train into a Library (ml-engineer)

### 2.1 What to extract

`csv_train.cpp` currently contains everything in a single translation unit with `main()` at line 3580. The extraction creates a companion library (`train_api.h` / `train_api.cpp`) that exposes the training codepath as a callable function, while `csv_train.cpp`'s `main()` becomes a thin CLI wrapper.

The key functions to extract from `csv_train.cpp`:

| Function | Current line | Extraction action |
|---|---|---|
| `TConfig` struct | 110–160 | Move to `train_api.h`, rename `TTrainConfig` |
| `TDataset` struct | 231–246 | Move to `train_api.h` |
| `TQuantization` struct | 621–627 | Move to `train_api.h` |
| `TPackedData` struct | 737–742 | Move to `train_api.h` |
| `TTreeRecord` struct | 1947–1957 | Move to `train_api.h` |
| `TTrainResult` struct | 2167–2176 | Move to `train_api.h` |
| `QuantizeFeatures()` | 629–702 | Move to `train_api.cpp` |
| `PackFeatures()` | 744–787 | Move to `train_api.cpp` |
| `CalcBasePrediction()` | 2460–2540 | Move to `train_api.cpp` |
| `ComputeLossValue()` | 1342–1430 | Move to `train_api.cpp` |
| `RunTraining()` | 2542–3574 | Move to `train_api.cpp` |
| `SaveModelJSON()` | 1975–2161 | Keep signature but add `SaveModelJSONToString()` overload |
| Feature importance loop | 4042–4072 | Extract into `CalcFeatureImportance()` |
| `TrainFromArrays()` | (new) | New entry point, described below |

### 2.2 `TTrainConfig` struct

Maps all 31 fields from the existing `TConfig` struct (lines 110–160). Rename to `TTrainConfig` to avoid ambiguity. All defaults must match `TConfig` exactly.

```cpp
// catboost/mlx/train_api.h

#pragma once
#include <string>
#include <vector>
#include <unordered_set>

using ui32 = uint32_t;

struct TTrainConfig {
    // Core training parameters
    ui32 NumIterations       = 100;
    ui32 MaxDepth            = 6;
    float LearningRate       = 0.1f;
    float L2RegLambda        = 3.0f;
    ui32 MaxBins             = 255;
    std::string LossType     = "auto";

    // Column roles (0-based column indices in the incoming feature matrix)
    // For TrainFromArrays(), TargetCol is unused (target is passed separately).
    // GroupCol and WeightCol index into the FEATURE matrix (not counting target).
    // -1 means "not present".
    int GroupCol             = -1;
    int WeightCol            = -1;   // unused by TrainFromArrays; weights passed separately

    // Categorical features
    std::unordered_set<int> CatFeatureCols;   // 0-based feature indices (not column indices)

    // Validation / early stopping
    float EvalFraction       = 0.0f;
    ui32 EarlyStoppingPatience = 0;

    // Subsampling
    float SubsampleRatio     = 1.0f;
    float ColsampleByTree    = 1.0f;
    ui32 RandomSeed          = 42;
    float RandomStrength     = 1.0f;

    // Bootstrap
    std::string BootstrapType     = "no";    // "no", "bayesian", "bernoulli", "mvs"
    float BaggingTemperature      = 1.0f;
    float MvsReg                  = 0.0f;

    // NaN handling
    std::string NanMode      = "min";         // "min" or "forbidden"

    // CTR target encoding
    bool UseCtr              = false;
    float CtrPrior           = 0.5f;
    ui32 MaxOneHotSize       = 10;

    // Regularization
    ui32 MinDataInLeaf       = 1;
    std::vector<int> MonotoneConstraints;     // per-feature: 0=none, 1=inc, -1=dec

    // Grow policy
    std::string GrowPolicy   = "SymmetricTree";   // "SymmetricTree", "Depthwise", "Lossguide"
    ui32 MaxLeaves           = 31;

    // Snapshot save/resume
    std::string SnapshotPath;
    ui32 SnapshotInterval    = 1;

    // Output control
    bool Verbose             = false;
    bool ComputeFeatureImportance = false;
};
```

### 2.3 `TTrainResult` struct

Extends the existing `TTrainResult` (lines 2167–2176) with the model JSON and feature importance map that the binding layer needs.

```cpp
// catboost/mlx/train_api.h (continued)

struct TTrainResult {
    // Existing fields (from RunTraining)
    std::vector<TTreeRecord>  Trees;
    float FinalTrainLoss      = 0.0f;
    float FinalTestLoss       = 0.0f;
    ui32  BestIteration       = 0;
    ui32  TreesBuilt          = 0;
    std::vector<float> BasePrediction;

    // Timing (ms, accumulated across all iterations)
    double GradMs             = 0;
    double TreeSearchMs       = 0;
    double LeafMs             = 0;
    double ApplyMs            = 0;

    // New fields for Python binding
    std::string ModelJSON;                        // complete JSON string (not written to disk)
    std::vector<std::string> FeatureNames;        // echoed back for Python to use
    std::unordered_map<std::string, double> FeatureImportance;  // name → gain
    std::vector<float> TrainLossHistory;          // per-iteration train loss
    std::vector<float> EvalLossHistory;           // per-iteration eval loss (empty if no val)
};
```

### 2.4 `TrainFromArrays()` function signature

This is the single new public entry point. It accepts numpy-friendly flat arrays (the binding layer converts numpy → these types) and returns a complete `TTrainResult`.

```cpp
// catboost/mlx/train_api.h (continued)

// Forward declare internal types used in function signatures.
struct TDataset;
struct TQuantization;

/// Train a GBDT model from in-memory arrays.
///
/// @param features       Row-major float32 matrix: [numDocs * numFeatures].
///                       Categorical features are pre-encoded as integer hash indices.
/// @param targets        Float32 vector: [numDocs].
/// @param featureNames   Names for each feature column. Length must equal numFeatures.
///                       Pass empty vector to use "f0", "f1", ... defaults.
/// @param isCategorical  Bool mask: [numFeatures]. true for categorical columns.
///                       Must agree with CatFeatureCols in config.
/// @param weights        Optional per-sample weights: [numDocs]. Empty = uniform.
/// @param groupIds       Optional group IDs for ranking losses: [numDocs]. Empty = no groups.
/// @param catHashMaps    Per-feature string→bin maps for categorical features.
///                       Must be pre-built by the caller to allow predict-time lookup.
///                       Length must equal numFeatures; empty map for numeric features.
/// @param numDocs        Number of training documents.
/// @param numFeatures    Number of feature columns.
/// @param valFeatures    Optional validation features (row-major): [valDocs * numFeatures].
/// @param valTargets     Optional validation targets: [valDocs]. Empty = no explicit val set.
/// @param valDocs        Number of validation documents (0 if no val set).
/// @param config         All training hyperparameters.
///
/// @returns TTrainResult with Trees, ModelJSON, FeatureImportance, and loss histories.
///          ModelJSON is a complete catboost-mlx-json string ready for json.loads().
///
TTrainResult TrainFromArrays(
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
```

### 2.5 `SaveModelJSONToString()` overload

Add alongside the existing `SaveModelJSON()` (line 1975). The existing function writes to a `FILE*`; this overload returns a `std::string`. The file-writing version in `csv_train.cpp` keeps its original signature for CLI use.

```cpp
// catboost/mlx/train_api.h (continued)

std::string SaveModelJSONToString(
    const std::vector<TTreeRecord>& allTrees,
    const TDataset&                 ds,
    const TQuantization&            quant,
    const std::string&              lossType,
    float                           lossParam,
    const TTrainConfig&             config,
    ui32                            approxDim,
    ui32                            numClasses,
    const std::vector<TCtrFeature>& ctrFeatures,
    const std::vector<float>&       basePrediction
);
```

### 2.6 `CalcFeatureImportance()` helper

Extract the gain-accumulation loop at lines 4042–4072 from `main()`:

```cpp
// catboost/mlx/train_api.h (continued)

/// Compute gain-based feature importance from a trained model.
/// Returns a map from feature name to total gain across all trees.
std::unordered_map<std::string, double> CalcFeatureImportance(
    const std::vector<TTreeRecord>& trees,
    const std::vector<std::string>& featureNames,
    uint32_t                        numFeatures
);
```

### 2.7 Refactored `main()` in csv_train.cpp

After extraction, `main()` becomes approximately 30 lines:

```cpp
// catboost/mlx/tests/csv_train.cpp  (after refactor)
// ... (all #includes and helper types remain, or #include "train_api.h")

int main(int argc, char** argv) {
    TConfig cliConfig = ParseArgs(argc, argv);
    TTrainConfig config = CliConfigToTrainConfig(cliConfig);  // thin adapter

    // Load data (CSV or binary format — CSV loading stays in csv_train.cpp)
    auto ds = IsBinaryFormat(cliConfig.CsvPath)
        ? LoadBinary(cliConfig.CsvPath, cliConfig.NanMode)
        : LoadCSV(cliConfig.CsvPath, cliConfig.TargetCol,
                  cliConfig.CatFeatureCols, cliConfig.NanMode,
                  cliConfig.GroupCol, cliConfig.WeightCol);

    printf("Loaded: %u rows, %u features\n", ds.NumDocs, ds.NumFeatures);

    // Flatten dataset into the array form TrainFromArrays expects
    std::vector<float> flatFeatures(ds.NumDocs * ds.NumFeatures);
    for (uint32_t f = 0; f < ds.NumFeatures; ++f)
        for (uint32_t d = 0; d < ds.NumDocs; ++d)
            flatFeatures[d * ds.NumFeatures + f] = ds.Features[f][d];

    auto result = TrainFromArrays(
        flatFeatures.data(), ds.Targets.data(),
        ds.FeatureNames, ds.IsCategorical,
        ds.Weights, ds.GroupIds, ds.CatHashMaps,
        ds.NumDocs, ds.NumFeatures,
        nullptr, nullptr, 0,   // no explicit val set (handled by EvalFraction inside)
        config
    );

    // CLI-specific output: save to file and print feature importance
    if (!cliConfig.OutputModelPath.empty() && !result.Trees.empty()) {
        // Write result.ModelJSON to file
        FILE* f = fopen(cliConfig.OutputModelPath.c_str(), "w");
        if (f) { fputs(result.ModelJSON.c_str(), f); fclose(f); }
    }
    if (cliConfig.ShowFeatureImportance) {
        // Print result.FeatureImportance in tabular format
        // ... (existing printf loop, unchanged)
    }
    return 0;
}
```

**Important:** The existing `LoadCSV()`, `LoadBinary()`, `ParseArgs()`, `IsBinaryFormat()`, and eval-file handling stay in `csv_train.cpp`. `TrainFromArrays()` receives arrays only — it has no file I/O.

---

## 3. Phase 2 — Nanobind Module (ml-engineer)

### 3.1 File location

```
python/catboost_mlx/_core/bindings.cpp
```

### 3.2 NB_MODULE structure

Pattern taken from `/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/mlx/python/src/mlx.cpp`:

```cpp
// python/catboost_mlx/_core/bindings.cpp

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/ndarray.h>

#include "catboost/mlx/train_api.h"

namespace nb = nanobind;

// Convenience alias for a read-only, C-contiguous, CPU float32 ndarray.
// Matches the pattern in mlx/python/src/convert.cpp.
using FloatArray = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Float1DArray = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using UInt32Array = nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

NB_MODULE(_core, m) {
    m.doc() = "CatBoost-MLX in-process training and prediction bindings.";

    // ── TTrainConfig binding ────────────────────────────────────────────────
    nb::class_<TTrainConfig>(m, "TrainConfig")
        .def(nb::init<>())
        .def_rw("num_iterations",        &TTrainConfig::NumIterations)
        .def_rw("max_depth",             &TTrainConfig::MaxDepth)
        .def_rw("learning_rate",         &TTrainConfig::LearningRate)
        .def_rw("l2_reg_lambda",         &TTrainConfig::L2RegLambda)
        .def_rw("max_bins",              &TTrainConfig::MaxBins)
        .def_rw("loss_type",             &TTrainConfig::LossType)
        .def_rw("group_col",             &TTrainConfig::GroupCol)
        .def_rw("eval_fraction",         &TTrainConfig::EvalFraction)
        .def_rw("early_stopping_patience", &TTrainConfig::EarlyStoppingPatience)
        .def_rw("subsample_ratio",       &TTrainConfig::SubsampleRatio)
        .def_rw("colsample_by_tree",     &TTrainConfig::ColsampleByTree)
        .def_rw("random_seed",           &TTrainConfig::RandomSeed)
        .def_rw("random_strength",       &TTrainConfig::RandomStrength)
        .def_rw("bootstrap_type",        &TTrainConfig::BootstrapType)
        .def_rw("bagging_temperature",   &TTrainConfig::BaggingTemperature)
        .def_rw("mvs_reg",              &TTrainConfig::MvsReg)
        .def_rw("nan_mode",              &TTrainConfig::NanMode)
        .def_rw("use_ctr",               &TTrainConfig::UseCtr)
        .def_rw("ctr_prior",             &TTrainConfig::CtrPrior)
        .def_rw("max_onehot_size",       &TTrainConfig::MaxOneHotSize)
        .def_rw("min_data_in_leaf",      &TTrainConfig::MinDataInLeaf)
        .def_rw("monotone_constraints",  &TTrainConfig::MonotoneConstraints)
        .def_rw("grow_policy",           &TTrainConfig::GrowPolicy)
        .def_rw("max_leaves",            &TTrainConfig::MaxLeaves)
        .def_rw("snapshot_path",         &TTrainConfig::SnapshotPath)
        .def_rw("snapshot_interval",     &TTrainConfig::SnapshotInterval)
        .def_rw("verbose",               &TTrainConfig::Verbose)
        .def_rw("compute_feature_importance", &TTrainConfig::ComputeFeatureImportance);

    // ── TTrainResult binding ────────────────────────────────────────────────
    nb::class_<TTrainResult>(m, "TrainResult")
        .def_ro("final_train_loss",      &TTrainResult::FinalTrainLoss)
        .def_ro("final_test_loss",       &TTrainResult::FinalTestLoss)
        .def_ro("best_iteration",        &TTrainResult::BestIteration)
        .def_ro("trees_built",           &TTrainResult::TreesBuilt)
        .def_ro("model_json",            &TTrainResult::ModelJSON)
        .def_ro("feature_importance",    &TTrainResult::FeatureImportance)
        .def_ro("train_loss_history",    &TTrainResult::TrainLossHistory)
        .def_ro("eval_loss_history",     &TTrainResult::EvalLossHistory)
        .def_ro("feature_names",         &TTrainResult::FeatureNames)
        .def_ro("grad_ms",               &TTrainResult::GradMs)
        .def_ro("tree_search_ms",        &TTrainResult::TreeSearchMs)
        .def_ro("leaf_ms",               &TTrainResult::LeafMs)
        .def_ro("apply_ms",              &TTrainResult::ApplyMs);

    // ── train() binding ─────────────────────────────────────────────────────
    m.def(
        "train",
        [](FloatArray features,          // shape [n_docs, n_features], float32, C-contiguous
           Float1DArray targets,          // shape [n_docs], float32
           const std::vector<std::string>& feature_names,
           const std::vector<bool>&        is_categorical,
           nb::object                      weights_obj,   // None or Float1DArray
           nb::object                      group_ids_obj, // None or UInt32Array
           const std::vector<std::unordered_map<std::string, uint32_t>>& cat_hash_maps,
           nb::object                      val_features_obj, // None or FloatArray
           nb::object                      val_targets_obj,  // None or Float1DArray
           const TTrainConfig&             config)
        -> TTrainResult
        {
            uint32_t n_docs = static_cast<uint32_t>(features.shape(0));
            uint32_t n_features = static_cast<uint32_t>(features.shape(1));

            const float* feat_ptr = features.data();
            const float* tgt_ptr  = targets.data();

            // Optional weights
            std::vector<float> weights_vec;
            if (!weights_obj.is_none()) {
                auto w = nb::cast<Float1DArray>(weights_obj);
                weights_vec.assign(w.data(), w.data() + w.size());
            }

            // Optional group IDs
            std::vector<uint32_t> group_ids_vec;
            if (!group_ids_obj.is_none()) {
                auto g = nb::cast<UInt32Array>(group_ids_obj);
                group_ids_vec.assign(g.data(), g.data() + g.size());
            }

            // Optional validation set
            const float* val_feat_ptr = nullptr;
            const float* val_tgt_ptr  = nullptr;
            uint32_t val_docs = 0;
            std::vector<float> val_feat_storage, val_tgt_storage;
            if (!val_features_obj.is_none() && !val_targets_obj.is_none()) {
                auto vf = nb::cast<FloatArray>(val_features_obj);
                auto vt = nb::cast<Float1DArray>(val_targets_obj);
                val_feat_ptr = vf.data();
                val_tgt_ptr  = vt.data();
                val_docs = static_cast<uint32_t>(vt.size());
            }

            TTrainResult result;
            // Release the GIL while the Metal GPU training runs.
            // All Metal/MLX calls are thread-safe; Python objects are not
            // accessed during training.
            {
                nb::gil_scoped_release release;
                result = TrainFromArrays(
                    feat_ptr, tgt_ptr,
                    feature_names, is_categorical,
                    weights_vec, group_ids_vec, cat_hash_maps,
                    n_docs, n_features,
                    val_feat_ptr, val_tgt_ptr, val_docs,
                    config
                );
            }
            return result;
        },
        nb::arg("features"),
        nb::arg("targets"),
        nb::arg("feature_names"),
        nb::arg("is_categorical"),
        nb::arg("weights")       = nb::none(),
        nb::arg("group_ids")     = nb::none(),
        nb::arg("cat_hash_maps") = std::vector<std::unordered_map<std::string, uint32_t>>{},
        nb::arg("val_features")  = nb::none(),
        nb::arg("val_targets")   = nb::none(),
        nb::arg("config")        = TTrainConfig{},
        "Train a GBDT model on Apple Silicon GPU. Releases the GIL during training."
    );
}
```

### 3.3 GIL release rationale

The C++ training loop calls Metal GPU kernels through MLX. No Python objects are accessed after the GIL is released. `nb::gil_scoped_release` is the nanobind analog of `py::gil_scoped_release` in pybind11 and is used identically. The GIL is automatically re-acquired when the `release` destructor runs at the closing `}`.

### 3.4 ndarray contract

Nanobind's `nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>` enforces at bind time:
- dtype must be float32
- array must be 2-dimensional (for features) or 1-dimensional (for targets/weights)
- memory must be C-contiguous (row-major)
- data must be on CPU

When `numpy.ascontiguousarray(X.astype(numpy.float32))` is called in Python before passing to `train()`, all four conditions are guaranteed.

### 3.5 `cat_hash_maps` parameter

For categorical features, `TrainFromArrays()` needs the string→bin index maps that `LoadCSV()` builds at lines 426–434 of `csv_train.cpp`. In the binding path, Python pre-encodes categorical strings to integer indices (matching the existing `_format_cat_col` behavior in core.py) and passes the corresponding hash maps. For a purely numeric dataset, pass an empty vector. This is the same information the subprocess path was encoding as hash-map JSON in the model file.

---

## 4. Phase 3 — Build System (devops-engineer)

### 4.1 Directory structure

```
python/
  catboost_mlx/
    _core/
      CMakeLists.txt     ← new
      bindings.cpp       ← new (Phase 2)
      __init__.py        ← new (re-exports TrainConfig, TrainResult, train)
  setup.py               ← new (replaces pure-setuptools build)
  pyproject.toml         ← modified
```

### 4.2 `python/catboost_mlx/_core/CMakeLists.txt`

Full content (modeled after `/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/mlx/python/src/CMakeLists.txt` and the MLX top-level FetchContent pattern at line 348):

```cmake
cmake_minimum_required(VERSION 3.27)
project(catboost_mlx_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Locate Python ─────────────────────────────────────────────────────────────
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# ── Fetch nanobind v2.10.2 ────────────────────────────────────────────────────
# Same tag used by MLX itself (mlx/CMakeLists.txt line 351).
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v2.10.2
  GIT_SHALLOW    TRUE
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(nanobind)

# ── Locate MLX ────────────────────────────────────────────────────────────────
# MLX_DIR is set by mlx.extension.CMakeBuild (extension.py line 63):
#   os.environ["MLX_DIR"] = mlx.__path__[0]
find_package(MLX REQUIRED PATHS "$ENV{MLX_DIR}/lib/cmake/mlx" NO_DEFAULT_PATH)

# ── Locate Apple frameworks ───────────────────────────────────────────────────
find_library(METAL_FRAMEWORK  Metal  REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# ── Locate CatBoost-MLX kernel headers ───────────────────────────────────────
# The extension is built from the catboost-mlx project root. Point at the
# headers needed by train_api.cpp.
if(NOT DEFINED CATBOOST_MLX_ROOT)
  # When invoked by setup.py the source tree root is two levels above this file.
  get_filename_component(CATBOOST_MLX_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
    ABSOLUTE)
endif()

# ── Build the nanobind extension ──────────────────────────────────────────────
nanobind_add_module(
  _core
  NB_STATIC
  STABLE_ABI
  LTO
  NB_DOMAIN  mlx
  "${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp"
  "${CATBOOST_MLX_ROOT}/catboost/mlx/train_api.cpp"
)

target_include_directories(_core PRIVATE
  "${CATBOOST_MLX_ROOT}"
)

target_link_libraries(_core PRIVATE
  mlx
  ${METAL_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
)

# macOS ARM64 only — enforce the deployment target
if(APPLE)
  set_target_properties(_core PROPERTIES
    OSX_DEPLOYMENT_TARGET "13.0"
  )
endif()

# Place the built .so alongside the Python package
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)
```

### 4.3 Changes to `python/pyproject.toml`

Add the build requirements for the CMake extension. The existing setuptools backend is replaced by a `setup.py`-driven build (standard pattern for CMake extensions, identical to MLX's own extension example).

```toml
[build-system]
requires = [
    "setuptools>=64",
    "wheel",
    "cmake>=3.27",
    "nanobind==2.10.2",
    "mlx>=0.18",
]
build-backend = "setuptools.build_meta"

[project]
name = "catboost-mlx"
version = "0.3.0"
# ... (rest of [project] unchanged) ...

[tool.setuptools.packages.find]
include = ["catboost_mlx*"]

[tool.setuptools.package-data]
catboost_mlx = ["bin/*", "py.typed", "_core/*.so", "_core/*.pyd"]
```

The only new entries in `[build-system].requires` are `cmake>=3.27`, `nanobind==2.10.2`, and bumping the minimum MLX to `>=0.18` (which ships `MLXConfig.cmake` required by `find_package(MLX)`).

### 4.4 `python/setup.py`

Uses `mlx.extension.CMakeExtension` and `mlx.extension.CMakeBuild` directly, identical to the pattern from `/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/mlx/python/mlx/extension.py`:

```python
# python/setup.py

from setuptools import setup
from mlx.extension import CMakeExtension, CMakeBuild

setup(
    ext_modules=[
        CMakeExtension(
            name="catboost_mlx._core",
            sourcedir="catboost_mlx/_core",
        )
    ],
    cmdclass={"build_ext": CMakeBuild},
)
```

`CMakeBuild.build_extension()` in `mlx/extension.py` (line 26) handles:
- Setting `CMAKE_LIBRARY_OUTPUT_DIRECTORY` to the correct output path
- Setting `MLX_DIR` environment variable from `mlx.__path__[0]`
- Forwarding `CMAKE_ARGS` from the environment
- Respecting `ARCHFLAGS` for cross-compilation
- Parallel builds via `CMAKE_BUILD_PARALLEL_LEVEL`

### 4.5 `python/catboost_mlx/_core/__init__.py`

```python
# python/catboost_mlx/_core/__init__.py
# Re-export the nanobind extension symbols so callers can write:
#   from catboost_mlx._core import TrainConfig, TrainResult, train

from ._core import TrainConfig, TrainResult, train  # noqa: F401

__all__ = ["TrainConfig", "TrainResult", "train"]
```

---

## 5. Phase 4 — Python Integration (ml-engineer)

### 5.1 Import fallback pattern

Add to the top of `python/catboost_mlx/core.py`, after the existing imports:

```python
# Attempt to import the nanobind in-process extension.
# Falls back to subprocess if the extension is not compiled.
try:
    from . import _core as _nb_core
    _HAS_NANOBIND = True
except ImportError:
    _nb_core = None
    _HAS_NANOBIND = False
```

### 5.2 `_build_train_config()` method

Add to `CatBoostMLX`. Maps all 31 Python params to `TTrainConfig` fields. Called only when `_HAS_NANOBIND` is True.

```python
def _build_train_config(self, cat_feature_indices: List[int]) -> "_nb_core.TrainConfig":
    """Map Python hyperparameters to a TTrainConfig for the nanobind path."""
    cfg = _nb_core.TrainConfig()
    cfg.num_iterations          = self.iterations
    cfg.max_depth               = self.depth
    cfg.learning_rate           = self.learning_rate
    cfg.l2_reg_lambda           = self.l2_reg_lambda
    cfg.max_bins                = self.bins
    cfg.loss_type               = _normalize_loss_str(self.loss)
    cfg.eval_fraction           = self.eval_fraction
    cfg.early_stopping_patience = self.early_stopping_rounds
    cfg.subsample_ratio         = self.subsample
    cfg.colsample_by_tree       = self.colsample_bytree
    cfg.random_seed             = self.random_seed
    cfg.random_strength         = self.random_strength
    cfg.bootstrap_type          = self.bootstrap_type
    cfg.bagging_temperature     = self.bagging_temperature
    cfg.mvs_reg                 = self.mvs_reg
    cfg.nan_mode                = self.nan_mode
    cfg.use_ctr                 = self.ctr
    cfg.ctr_prior               = self.ctr_prior
    cfg.max_onehot_size         = self.max_onehot_size
    cfg.min_data_in_leaf        = self.min_data_in_leaf
    cfg.monotone_constraints    = list(self.monotone_constraints or [])
    cfg.grow_policy             = self.grow_policy or "SymmetricTree"
    cfg.max_leaves              = self.max_leaves
    cfg.snapshot_path           = self.snapshot_path or ""
    cfg.snapshot_interval       = self.snapshot_interval
    cfg.verbose                 = self.verbose
    cfg.compute_feature_importance = True  # always compute; Python filters display
    # cat_feature_indices are set in the C++ call as the isCategorical mask
    return cfg
```

### 5.3 `_fit_nanobind()` method

Add to `CatBoostMLX`. Replaces Phases 5–8 of `fit()` when `_HAS_NANOBIND` is True:

```python
def _fit_nanobind(
    self, X: np.ndarray, y: np.ndarray,
    feature_names: List[str],
    cat_features: Optional[List[int]],
    weights: Optional[np.ndarray],
    group_ids: Optional[np.ndarray],
    val_X: Optional[np.ndarray],
    val_y: Optional[np.ndarray],
) -> None:
    """Run training via the nanobind in-process path."""
    n_docs, n_features = X.shape

    # Ensure correct dtypes and C-contiguous layout (nanobind enforces this)
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)
    y_f32 = np.ascontiguousarray(y, dtype=np.float32)

    # Build is_categorical mask and cat_hash_maps
    cat_set = set(cat_features) if cat_features else set()
    is_categorical = [bool(f in cat_set) for f in range(n_features)]

    # For categorical features, build string→bin hash maps.
    # X must already contain integer-encoded categorical indices
    # (core.py's _format_cat_col handles string→int encoding before this call).
    cat_hash_maps = [{} for _ in range(n_features)]  # populated below

    # Build TTrainConfig
    cfg = self._build_train_config(list(cat_set))

    # Optional arrays
    w_arr = np.ascontiguousarray(weights, dtype=np.float32) if weights is not None else None
    g_arr = np.ascontiguousarray(group_ids, dtype=np.uint32) if group_ids is not None else None

    val_X_f32 = np.ascontiguousarray(val_X, dtype=np.float32) if val_X is not None else None
    val_y_f32 = np.ascontiguousarray(val_y, dtype=np.float32) if val_y is not None else None

    result = _nb_core.train(
        features=X_f32,
        targets=y_f32,
        feature_names=feature_names,
        is_categorical=is_categorical,
        weights=w_arr,
        group_ids=g_arr,
        cat_hash_maps=cat_hash_maps,
        val_features=val_X_f32,
        val_targets=val_y_f32,
        config=cfg,
    )

    # Parse result back into self state (same fields as subprocess path)
    self._model_data = json.loads(result.model_json)
    self._train_loss_history = list(result.train_loss_history)
    self._eval_loss_history  = list(result.eval_loss_history)
    self._feature_importance = dict(result.feature_importance)
    self._is_fitted = True
    self._model_json_cache = None

    # Inject real feature names into model data (matches subprocess path behavior)
    for i, feat in enumerate(self._model_data.get("features", [])):
        if i < len(feature_names):
            feat["name"] = feature_names[i]

    # Persist cat_features in model_info for save/load roundtrips
    info = self._model_data.get("model_info", {})
    info["cat_features"] = cat_features
```

### 5.4 Integration point in `fit()`

In the existing `fit()` method, replace Phases 5–8 (lines 1090–1203) with a dispatch:

```python
# ── Phase 5: Set sklearn-required attributes ──
self.n_features_in_ = X.shape[1]
self.n_outputs_ = 1
names = feature_names or [f"f{i}" for i in range(X.shape[1])]
self.feature_names_in_ = np.array(names, dtype=object)

# ── Phase 6: Route to nanobind or subprocess ──
if _HAS_NANOBIND:
    val_X_nb, val_y_nb = None, None
    if eval_set is not None:
        val_X_nb = _to_numpy(eval_set[0])
        val_y_nb = _to_numpy(eval_set[1])
    self._fit_nanobind(X, y, names, self.cat_features,
                       sw, gid, val_X_nb, val_y_nb)
else:
    # Original subprocess path (unchanged from current implementation)
    self._fit_subprocess(X, y, names, sw, gid, eval_set)
```

The existing `fit()` body from lines 1090 onward moves into a `_fit_subprocess()` method. No logic changes.

### 5.5 Backward compatibility

The following behaviors are preserved unchanged:

- `save_model()` / `load_model()` read and write `self._model_data` as JSON — the format is identical since `ModelJSON` is produced by `SaveModelJSONToString()` using the same serialization code as `SaveModelJSON()`.
- `predict()` uses `self._model_data` via the Python-side tree walker (`_predict_utils.py`) — no change.
- `get_feature_importance()` reads `self._feature_importance` — now populated directly from `result.feature_importance` instead of parsed from stdout regex.
- `get_train_loss_history()` / `get_eval_loss_history()` — populated from `result.train_loss_history` / `result.eval_loss_history`.
- `cross_validate()` — can continue using subprocess path; nanobind path doesn't need to support it in Sprint 11.

---

## 6. Phase 5 — Testing (qa-engineer)

### 6.1 Parity test: subprocess vs nanobind

Create `python/tests/test_nanobind_parity.py`. The test trains identical models via both code paths and asserts numerical agreement.

```python
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris

from catboost_mlx import CatBoostMLXClassifier, CatBoostMLXRegressor
from catboost_mlx import _core as _nb_core  # skip test if not available

pytestmark = pytest.mark.skipif(
    _nb_core is None,
    reason="nanobind extension not compiled"
)

SEED = 42
TOL_LOSS = 1e-4    # acceptable difference in final train loss
TOL_PRED = 1e-5    # acceptable element-wise difference in predictions

def _train_both_paths(estimator_cls, X, y, **kwargs):
    """Return (nanobind_model, subprocess_model) trained on the same data."""
    import catboost_mlx.core as core_mod

    model_nb = estimator_cls(random_seed=SEED, **kwargs)
    model_sb = estimator_cls(random_seed=SEED, **kwargs)

    model_nb.fit(X, y)                    # uses nanobind if available

    # Force subprocess path by temporarily disabling nanobind
    orig = core_mod._HAS_NANOBIND
    core_mod._HAS_NANOBIND = False
    try:
        model_sb.fit(X, y)
    finally:
        core_mod._HAS_NANOBIND = orig

    return model_nb, model_sb


def test_parity_regression():
    rng = np.random.RandomState(SEED)
    X = rng.randn(200, 5).astype(np.float32)
    y = X[:, 0] * 2 + rng.randn(200).astype(np.float32) * 0.1

    nb_model, sb_model = _train_both_paths(
        CatBoostMLXRegressor, X, y, iterations=50, depth=4
    )

    assert abs(nb_model._train_loss_history[-1] -
               sb_model._train_loss_history[-1]) < TOL_LOSS

    nb_preds = nb_model.predict(X)
    sb_preds = sb_model.predict(X)
    np.testing.assert_allclose(nb_preds, sb_preds, atol=TOL_PRED)


def test_parity_binary_classification():
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)

    nb_model, sb_model = _train_both_paths(
        CatBoostMLXClassifier, X, y, iterations=50, loss="logloss"
    )

    assert abs(nb_model._train_loss_history[-1] -
               sb_model._train_loss_history[-1]) < TOL_LOSS


def test_parity_multiclass():
    X, y = load_iris(return_X_y=True)
    X = X.astype(np.float32)

    nb_model, sb_model = _train_both_paths(
        CatBoostMLXClassifier, X, y, iterations=50, loss="multiclass"
    )

    nb_preds = nb_model.predict(X)
    sb_preds = sb_model.predict(X)
    np.testing.assert_array_equal(nb_preds, sb_preds)


def test_parity_validation_split():
    rng = np.random.RandomState(SEED)
    X = rng.randn(300, 8).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)

    nb_model, sb_model = _train_both_paths(
        CatBoostMLXClassifier, X, y,
        iterations=40, loss="logloss", eval_fraction=0.2
    )

    assert len(nb_model._eval_loss_history) > 0
    assert abs(nb_model._eval_loss_history[-1] -
               sb_model._eval_loss_history[-1]) < TOL_LOSS


def test_parity_early_stopping():
    rng = np.random.RandomState(SEED)
    X = rng.randn(300, 6).astype(np.float32)
    y = rng.randn(300).astype(np.float32)

    nb_model, sb_model = _train_both_paths(
        CatBoostMLXRegressor, X, y,
        iterations=100, eval_fraction=0.2, early_stopping_rounds=10
    )

    # Both paths should stop at the same iteration
    assert nb_model._model_data["model_info"]["num_trees"] == \
           sb_model._model_data["model_info"]["num_trees"]


def test_parity_lossguide():
    rng = np.random.RandomState(SEED)
    X = rng.randn(200, 5).astype(np.float32)
    y = rng.randn(200).astype(np.float32)

    nb_model, sb_model = _train_both_paths(
        CatBoostMLXRegressor, X, y,
        iterations=30, grow_policy="Lossguide", max_leaves=15
    )

    assert abs(nb_model._train_loss_history[-1] -
               sb_model._train_loss_history[-1]) < TOL_LOSS
```

### 6.2 Build verification test

Create `python/tests/test_nanobind_build.py`:

```python
"""Smoke test: confirm the extension loads and exposes expected symbols."""
import pytest

def test_extension_importable():
    try:
        from catboost_mlx import _core
    except ImportError as e:
        pytest.skip(f"nanobind extension not compiled: {e}")
    assert hasattr(_core, "TrainConfig")
    assert hasattr(_core, "TrainResult")
    assert hasattr(_core, "train")


def test_trainconfig_defaults():
    from catboost_mlx._core import TrainConfig
    cfg = TrainConfig()
    assert cfg.num_iterations == 100
    assert cfg.max_depth == 6
    assert abs(cfg.learning_rate - 0.1) < 1e-6
    assert cfg.grow_policy == "SymmetricTree"
    assert cfg.max_leaves == 31


def test_gil_released(tmp_path):
    """Confirm training completes without a GIL deadlock."""
    import threading
    import numpy as np

    try:
        from catboost_mlx._core import TrainConfig, train
    except ImportError:
        pytest.skip("nanobind extension not compiled")

    rng = np.random.RandomState(0)
    X = np.ascontiguousarray(rng.randn(100, 3), dtype=np.float32)
    y = np.ascontiguousarray(rng.randn(100), dtype=np.float32)

    results = []
    def run():
        cfg = TrainConfig()
        cfg.num_iterations = 10
        cfg.loss_type = "rmse"
        r = train(features=X, targets=y,
                  feature_names=["a", "b", "c"],
                  is_categorical=[False, False, False],
                  config=cfg)
        results.append(r.trees_built)

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "training thread hung — possible GIL deadlock"
    assert results[0] == 10
```

### 6.3 Edge case tests

These supplement the existing `python/tests/` suite and do not replace it:

| Test | What it verifies |
|---|---|
| `test_categorical_features_parity` | OneHot and CTR cats produce same predictions on both paths |
| `test_sample_weight_parity` | `sample_weight` propagates identically |
| `test_feature_importance_parity` | `feature_importance` dict keys and values match (atol=1e-3) |
| `test_save_load_roundtrip_nanobind` | `save_model()` / `load_model()` roundtrip after nanobind training |
| `test_predict_after_nanobind_fit` | `predict()` produces same output as after subprocess fit |
| `test_cross_validate_still_uses_subprocess` | `cross_validate()` falls through to subprocess path gracefully |

### 6.4 Build verification on macOS ARM64

Run as part of CI before any parity tests:

```bash
# From python/
pip install -e ".[dev]" --no-build-isolation
python -c "from catboost_mlx._core import train; print('extension OK')"
```

If the build fails, the parity tests auto-skip (the `skipif` decorator handles this). The subprocess path must still pass all existing tests regardless.

---

## 7. File Inventory

| File | Action | Owner | Notes |
|---|---|---|---|
| `catboost/mlx/train_api.h` | **Create** | ml-engineer | `TTrainConfig`, `TTrainResult`, `TrainFromArrays()`, `SaveModelJSONToString()`, `CalcFeatureImportance()` |
| `catboost/mlx/train_api.cpp` | **Create** | ml-engineer | Implementations of all functions in `train_api.h`; extracted from `csv_train.cpp` |
| `catboost/mlx/tests/csv_train.cpp` | **Modify** | ml-engineer | Thin `main()` wrapper; move functions to `train_api.cpp`; add `#include "catboost/mlx/train_api.h"` |
| `python/catboost_mlx/_core/bindings.cpp` | **Create** | ml-engineer | NB_MODULE, type bindings, `train()` function |
| `python/catboost_mlx/_core/CMakeLists.txt` | **Create** | devops-engineer | Full content in Section 4.2 |
| `python/catboost_mlx/_core/__init__.py` | **Create** | devops-engineer | Re-exports `TrainConfig`, `TrainResult`, `train` |
| `python/setup.py` | **Create** | devops-engineer | CMakeExtension + CMakeBuild setup |
| `python/pyproject.toml` | **Modify** | devops-engineer | Add `cmake>=3.27`, `nanobind==2.10.2`, `mlx>=0.18` to `[build-system].requires` |
| `python/catboost_mlx/core.py` | **Modify** | ml-engineer | Add `_HAS_NANOBIND` fallback, `_build_train_config()`, `_fit_nanobind()`, dispatch in `fit()`, refactor to `_fit_subprocess()` |
| `python/tests/test_nanobind_parity.py` | **Create** | qa-engineer | Parity tests (Section 6.1) |
| `python/tests/test_nanobind_build.py` | **Create** | qa-engineer | Build smoke tests (Section 6.2) |

---

## 8. Acceptance Criteria

1. `pip install -e ".[dev]"` completes without error on macOS ARM64 (Apple Silicon) with Xcode Command Line Tools installed.
2. `python -c "from catboost_mlx._core import TrainConfig, TrainResult, train; print('OK')"` prints `OK`.
3. `TrainConfig()` exposes all 31 fields listed in Section 2.2 with the correct default values.
4. `_core.train()` with a 200-row float32 dataset and `num_iterations=50` completes without crash or hang.
5. All parity tests in `test_nanobind_parity.py` pass (final train loss difference < 1e-4 between nanobind and subprocess paths).
6. All six edge case tests listed in Section 6.3 pass.
7. The GIL deadlock test in `test_nanobind_build.py` completes within 30 seconds.
8. The existing test suite (`pytest python/tests/` excluding new files) passes without regression.
9. `csv_train` binary still compiles and produces identical results to its pre-refactor behavior (`test_qa_round6.py` and `test_qa_round7.py` pass).
10. `save_model()` / `load_model()` roundtrip works after a nanobind-path fit (predictions are identical before and after serialization).
11. `CatBoostMLX(verbose=True).fit(X, y)` does not crash when the nanobind path is active (progress reporting may differ from PTY path, but must not error).
12. `cross_validate()` falls back gracefully to the subprocess path when called on a nanobind-trained model.

---

## 9. Risk Register

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| **Memory layout mismatch.** `csv_train.cpp` stores features as `[numFeatures][numDocs]` (column-major), but `TrainFromArrays()` receives a row-major `[numDocs, numFeatures]` numpy array. Transposing in the wrapper introduces a copy. | High | Certain | The spec explicitly specifies a transposition in the `csv_train.cpp` refactor (Section 2.7). Document the expected layout in `TrainFromArrays()`'s comment. Add an assertion `features.shape == (n_docs, n_features)` in the Python wrapper. |
| **MLXConfig.cmake not found.** `find_package(MLX)` fails if the installed MLX version predates the CMake package file. MLX shipped `MLXConfig.cmake` in 0.18. | High | Medium | Pin `mlx>=0.18` in `[build-system].requires`. `extension.py`'s `CMakeBuild` already sets `MLX_DIR`; document the `MLX_DIR` fallback path for developers with custom MLX builds. |
| **printf to stdout from C++ during nanobind call.** `TrainFromArrays()` calls `printf()` for progress reporting. With the GIL released, this prints to the process's stdout outside Python's control. Verbose output still works but cannot be captured by Python's `sys.stdout` redirect. | Medium | High | In Sprint 11, this is acceptable. For Sprint 12, replace `printf()` calls in `TrainFromArrays()` with a callback function pointer (type: `void (*progress_cb)(const char*)`) that the Python binding can set to a Python-side handler. Document this as known tech debt. |
| **GIL re-entrance during Metal callback.** If MLX fires a completion handler on the same thread that holds the GIL (possible if MLX uses a serial dispatch queue), `nb::gil_scoped_release` could deadlock. | Medium | Low | MLX's Metal command buffers complete on a dedicated Metal completion queue, not the Python main thread. Confirmed by reading `mlx/mlx/backend/metal/metal.cpp`. Add the GIL deadlock test (Section 6.2) to catch regressions. |
| **cat_hash_maps roundtrip.** The binding path passes empty `cat_hash_maps` in Sprint 11. Categorical feature predict-time lookup requires these maps (they are stored in the JSON model file by `SaveModelJSONToString()`). The JSON roundtrip preserves them; the issue is only if someone calls `TrainFromArrays()` directly and expects to use the maps at predict time. | Medium | Low | `predict()` uses the Python-side tree walker from `_predict_utils.py`, which reads `cat_hash_map` from the model JSON — not from `cat_hash_maps` passed to `train()`. No action needed in Sprint 11. Document for Sprint 12 if a C++ predict binding is added. |
| **nanobind version incompatibility.** nanobind's ABI is not stable across minor versions. Using v2.10.2 pinned in both `pyproject.toml` and `FetchContent_Declare` ensures consistency, but future MLX upgrades may require a different nanobind version. | Low | Low | Pin both to `v2.10.2` (same as MLX). When MLX upgrades nanobind, upgrade both together. The `NB_DOMAIN mlx` declaration (used in both MLX's `CMakeLists.txt` line 8 and our `CMakeLists.txt`) ensures our module and MLX can coexist in the same process without symbol conflicts. |
| **csv_train.cpp refactor regression.** Moving functions to `train_api.cpp` risks subtle behavior changes if any function has file-static state (e.g., Metal pipeline caches). | Medium | Low | Metal pipeline caches in MLX are process-global, not file-static. The `mx::fast::metal_kernel()` registry is a global map. Moving functions across translation units does not affect this. Run `test_qa_round6.py` and `test_qa_round7.py` after refactor to confirm. |
| **macOS version gating.** `OSX_DEPLOYMENT_TARGET 13.0` may exclude users on macOS 12 (Monterey) who have Apple Silicon but lack some Metal 3 features used by MLX. | Low | Low | MLX itself requires macOS 13.3+. Our deployment target matches. Document this in the user-facing error message when the extension fails to load. |
