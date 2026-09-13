# Native Metal training is part of the existing CatBoost CLI and Python extension.
# Metal shader sources stay embedded in native headers; wheels need no shader files.
if (NOT HAVE_METAL)
  return()
endif()

# Use the same C++ ABI and compilation settings as the vendored CatBoost runtime.
set(CMAKE_OBJCXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_OBJCXX_FLAGS}")
set(CMAKE_OBJCXX_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)

find_library(CATBOOST_METAL_FRAMEWORK Metal REQUIRED)
find_library(CATBOOST_FOUNDATION_FRAMEWORK Foundation REQUIRED)

add_library(catboost-metal-native STATIC
  native/metal_trainer.mm
  native/metal_ordered_trainer.mm
  native/metal_greedy_trainer.mm
  native/metal_multiclass.mm
  native/metal_inference.mm
  native/metal_ctrs.mm
  native/metal_sort.mm
  native/metal_projection.mm
)
target_compile_options(catboost-metal-native PRIVATE -fobjc-arc)
target_link_libraries(catboost-metal-native PUBLIC
  contrib-libs-cxxsupp
  ${CATBOOST_METAL_FRAMEWORK}
  ${CATBOOST_FOUNDATION_FRAMEWORK}
  objc
)

add_subdirectory(train_lib)
