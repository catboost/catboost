# CatBoost-MLX: Metal GPU backend for Apple Silicon
#
# This backend replaces CUDA for GPU-accelerated GBDT training on macOS ARM64.
# Activated by -DUSE_MLX=ON (mutually exclusive with HAVE_CUDA).

cmake_minimum_required(VERSION 3.20)

option(USE_MLX "Build the MLX Metal GPU backend for Apple Silicon" OFF)

# Per-stage profiling instrumentation (S16-01).
# When ON, compiles the CATBOOST_MLX_STAGE_PROFILE preprocessor macro into all
# training translation units.  The macro gates TStageProfiler scoped timers in
# mlx_boosting.cpp and structure_searcher.cpp — zero overhead when OFF because
# the macro expands to nothing and the dead-code path is eliminated by the compiler.
#
# Activate with: cmake -DUSE_MLX=ON -DCATBOOST_MLX_STAGE_PROFILE=ON ...
option(CATBOOST_MLX_STAGE_PROFILE
    "Enable per-stage pipeline profiling in the MLX training loop (default OFF)" OFF)

if (NOT USE_MLX)
    return()
endif()

if (HAVE_CUDA)
    message(FATAL_ERROR "USE_MLX and HAVE_CUDA are mutually exclusive. Disable CUDA to use the MLX backend.")
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(FATAL_ERROR "USE_MLX requires macOS on Apple Silicon (darwin-arm64).")
endif()

# Find MLX
set(MLX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../refs/mlx" CACHE PATH "Path to MLX source tree")
if (NOT EXISTS "${MLX_DIR}/CMakeLists.txt")
    # Try sibling directory layout
    set(MLX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../mlx" CACHE PATH "Path to MLX source tree" FORCE)
endif()

if (NOT EXISTS "${MLX_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "MLX not found. Set MLX_DIR to the MLX source root (tried: ${MLX_DIR})")
endif()

message(STATUS "CatBoost-MLX: Using MLX from ${MLX_DIR}")

# Find system Metal frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# --- MLX backend library ---

add_library(catboost-mlx-train-lib STATIC
    train_lib/train.cpp
    train_lib/model_exporter.cpp
    gpu_data/data_set_builder.cpp
    methods/histogram.cpp
    methods/score_calcer.cpp
    methods/structure_searcher.cpp
    methods/tree_applier.cpp
    methods/leaves/leaf_estimator.cpp
    methods/mlx_boosting.cpp
)

target_include_directories(catboost-mlx-train-lib PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${MLX_DIR}
)

target_link_libraries(catboost-mlx-train-lib PUBLIC
    ${METAL_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
)

# Ensure C++17 minimum (matching CatBoost)
target_compile_features(catboost-mlx-train-lib PUBLIC cxx_std_17)

# Propagate the profiling macro to all translation units that include stage_profiler.h.
# The macro is guarded with #ifdef in the header, so non-profiling builds see no overhead.
if (CATBOOST_MLX_STAGE_PROFILE)
    message(STATUS "CatBoost-MLX: Stage profiling ENABLED (CATBOOST_MLX_STAGE_PROFILE=ON)")
    target_compile_definitions(catboost-mlx-train-lib PUBLIC CATBOOST_MLX_STAGE_PROFILE)
endif()

# Add Objective-C++ support for Metal API calls
set_source_files_properties(
    train_lib/train.cpp
    PROPERTIES COMPILE_FLAGS "-x objective-c++"
)
