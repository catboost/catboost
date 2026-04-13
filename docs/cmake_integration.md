# CMake Integration Spec — CatBoost-MLX into Upstream CatBoost

> **Purpose:** Technical reference for how `catboost/mlx/` integrates with CatBoost's
> existing CMake build system. Written for the upstream CatBoost team's review.

---

## Overview

The MLX backend mirrors the CUDA backend's build pattern:

| Concept | CUDA | MLX |
|---------|------|-----|
| Feature flag | `HAVE_CUDA` | `HAVE_MLX` |
| Platform file | `CMakeLists.linux-x86_64-cuda.txt` | `CMakeLists.darwin-arm64.txt` (modified) |
| Source directory | `catboost/cuda/` | `catboost/mlx/` |
| External dependency | `find_package(CUDAToolkit)` | `find_package(MLX)` or `pkg_check_modules(MLX)` |
| Static registrator | `TTrainerFactory::TRegistrator<TGPUModelTrainer>` | `TTrainerFactory::TRegistrator<TMLXModelTrainer>` |
| Task type | `ETaskType::GPU` | `ETaskType::GPU` (same — mutually exclusive) |

**Key constraint:** `HAVE_MLX` and `HAVE_CUDA` are mutually exclusive. This is safe because CUDA is never available on darwin-arm64 and MLX is never available on Linux/Windows.

---

## Changes Required

### 1. Root `CMakeLists.txt` — Add `HAVE_MLX` option

```cmake
# After HAVE_CUDA option
option(HAVE_MLX "Build MLX (Metal GPU) backend for Apple Silicon" OFF)

# Auto-detect: enable MLX on darwin-arm64 when MLX is found and CUDA is off
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" AND NOT HAVE_CUDA)
    find_package(MLX QUIET)
    if (MLX_FOUND)
        set(HAVE_MLX ON)
        message(STATUS "MLX found — enabling Apple Silicon GPU backend")
    endif()
endif()

# Mutual exclusion guard
if (HAVE_MLX AND HAVE_CUDA)
    message(FATAL_ERROR "HAVE_MLX and HAVE_CUDA are mutually exclusive.")
endif()
```

### 2. `catboost/CMakeLists.darwin-arm64.txt` — Add MLX subdirectory

Current file lists `add_subdirectory(...)` for app, libs, private, etc. Add:

```cmake
# At the end of catboost/CMakeLists.darwin-arm64.txt
if (HAVE_MLX)
    add_subdirectory(mlx)
endif()
```

### 3. `catboost/mlx/CMakeLists.txt` — Backend library

Already exists. Key integration points:

```cmake
# catboost/mlx/CMakeLists.txt

# Static library for the MLX training backend
add_library(catboost-mlx-train-lib STATIC
    train_lib/train.cpp          # TMLXModelTrainer (IModelTrainer impl)
    train_lib/model_exporter.cpp # TFullModel export (symmetric + non-symmetric)
    gpu_data/data_set_builder.cpp
    methods/histogram.cpp
    methods/score_calcer.cpp
    methods/structure_searcher.cpp
    methods/tree_applier.cpp
    methods/leaves/leaf_estimator.cpp
    methods/mlx_boosting.cpp
)

target_link_libraries(catboost-mlx-train-lib PUBLIC
    MLX::mlx               # MLX C++ library
    "-framework Metal"
    "-framework Foundation"
    "-framework Accelerate"
)
```

### 4. Static registrator linkage

CatBoost uses `add_global_library_for` to ensure static registrators are linked into the final binary (not dead-stripped). The MLX backend needs the same treatment:

```cmake
# In the training binary / Python extension CMakeLists:
if (HAVE_MLX)
    add_global_library_for(catboost-mlx-train-lib)
endif()
```

This ensures `TTrainerFactory::TRegistrator<TMLXModelTrainer>` in `train_lib/train.cpp` is not stripped, so `ETaskType::GPU` resolves to the MLX trainer at runtime.

### 5. Python package integration

The Python package (`catboost/python-package/`) would need a conditional dependency:

```cmake
# In catboost/python-package/CMakeLists.txt
if (HAVE_MLX)
    target_link_libraries(catboost_python PRIVATE catboost-mlx-train-lib)
    target_compile_definitions(catboost_python PRIVATE HAVE_MLX=1)
endif()
```

With `HAVE_MLX=1` defined, the Python package's C++ extension can call `GetTrainerFactory().Construct(ETaskType::GPU, ...)` on darwin-arm64, and it will resolve to `TMLXModelTrainer`.

---

## MLX Dependency Resolution

MLX can be found via several paths:

| Method | Command | Notes |
|--------|---------|-------|
| Homebrew | `brew install mlx` | Ships headers + dylib |
| pip | `pip install mlx` | Ships headers + dylib in site-packages |
| Source build | `cmake --build` from mlx repo | Development builds |

**Recommended: `find_package(MLX)`** with a fallback to `pkg_check_modules`:

```cmake
find_package(MLX QUIET CONFIG)
if (NOT MLX_FOUND)
    # Fallback: try pkg-config or manual paths
    find_library(MLX_LIBRARY mlx HINTS /opt/homebrew/lib)
    find_path(MLX_INCLUDE_DIR mlx/mlx.h HINTS /opt/homebrew/include)
    if (MLX_LIBRARY AND MLX_INCLUDE_DIR)
        add_library(MLX::mlx SHARED IMPORTED)
        set_target_properties(MLX::mlx PROPERTIES
            IMPORTED_LOCATION ${MLX_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${MLX_INCLUDE_DIR}
        )
        set(MLX_FOUND TRUE)
    endif()
endif()
```

---

## ya.make Consideration

CatBoost's primary build system is YaTool (`ya.make`). We do not have access to YaTool and cannot provide `ya.make` files. Options:

1. **CatBoost team ports CMake → ya.make** (preferred — they know the build system)
2. **CMake-only for darwin-arm64** — since `ya.make` may not target macOS ARM64 anyway
3. **Dual maintenance** — we provide CMake, team adds ya.make if needed

---

## Files Added (no existing files modified)

```
catboost/mlx/
├── CMakeLists.txt              # Backend build (already exists)
├── kernels/
│   ├── hist.metal              # Histogram Metal shader
│   ├── scores.metal            # Split scoring shader
│   ├── apply.metal             # Tree application shader
│   ��── leaf.metal              # Leaf estimation shader
│   └── kernel_sources.h        # Inline Metal source strings
├── gpu_data/
│   ├── mlx_data_set.h          # GPU data layout
│   └── data_set_builder.cpp    # CPU → GPU transfer
├── methods/
│   ├── histogram.cpp/h
│   ├── score_calcer.cpp/h
│   ├── structure_searcher.cpp/h
│   ├── tree_applier.cpp/h
│   ├── leaves/leaf_estimator.cpp/h
│   └── mlx_boosting.cpp/h
├── targets/
��   ├── mlx_target.h            # IMLXTargetFunc interface
│   └── pairwise_target.h       # PairLogit / YetiRank
├── train_lib/
│   ├── train.h/cpp             # TMLXModelTrainer (IModelTrainer)
│   └── model_exporter.h/cpp    # TFullModel export
└── tests/
    ├── csv_train.cpp           # Standalone training binary
    ├��─ csv_predict.cpp         # Standalone prediction binary
    └── model_export_test.cpp   # Unit tests
```

**Zero modifications to existing CatBoost source files.** The only existing file that needs a one-line addition is `catboost/CMakeLists.darwin-arm64.txt` (add `add_subdirectory(mlx)` guarded by `HAVE_MLX`).
