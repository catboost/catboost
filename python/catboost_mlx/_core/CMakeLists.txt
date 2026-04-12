cmake_minimum_required(VERSION 3.27)
project(catboost_mlx_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Locate Python ─────────────────────────────────────────────────────────────
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# ── Locate nanobind (installed via pip) ───────────────────────────────────────
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ── Locate MLX ────────────────────────────────────────────────────────────────
# MLX_DIR is set by mlx.extension.CMakeBuild via os.environ["MLX_DIR"].
# The cmake config lives at $MLX_DIR/share/cmake/MLX/MLXConfig.cmake.
if(DEFINED ENV{MLX_DIR})
  set(MLX_DIR "$ENV{MLX_DIR}/share/cmake/MLX")
endif()
find_package(MLX REQUIRED)

# ── Apple frameworks ──────────────────────────────────────────────────────────
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# ── CatBoost-MLX project root ────────────────────────────────────────────────
# When invoked by setup.py the source tree root is three levels above _core/.
if(NOT DEFINED CATBOOST_MLX_ROOT)
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

# Suppress warnings from included csv_train.cpp
target_compile_options(_core PRIVATE
  -Wno-c++20-extensions
  -Wno-unused-function
)

# macOS ARM64 deployment target
if(APPLE)
  set_target_properties(_core PROPERTIES
    OSX_DEPLOYMENT_TARGET "13.0"
  )
endif()

# Place the built .so alongside the Python package
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)
