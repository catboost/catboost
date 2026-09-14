add_library(catboost-metal-train_lib INTERFACE)

set(METAL_TRAIN_LIB_DEPENDENCIES
  contrib-libs-cxxsupp
  yutil
  catboost-metal-native
  catboost-libs-train_lib
  catboost-libs-data
  catboost-libs-eval_result
  catboost-libs-helpers
  catboost-libs-loggers
  catboost-libs-logging
  catboost-libs-metrics
  catboost-libs-model
  catboost-libs-overfitting_detector
  private-libs-algo
  private-libs-algo_helpers
  private-libs-options
  private-libs-quantization
  private-libs-target
  library-cpp-json
  library-cpp-object_factory
  cpp-threading-local_executor
)
target_link_libraries(catboost-metal-train_lib INTERFACE ${METAL_TRAIN_LIB_DEPENDENCIES})

# Factory registration runs through static initializers. Retain it with the
# repository's existing Darwin -force_load implementation.
add_global_library_for(catboost-metal-train_lib.global catboost-metal-train_lib)
# -force_load is a linker option, so its archive is otherwise only an
# order-only dependency. Rebuild consumers when trainer/progress code changes.
set_property(TARGET catboost-metal-train_lib.global.wholearchive APPEND PROPERTY
  INTERFACE_LINK_DEPENDS "$<TARGET_FILE:catboost-metal-train_lib.global>")
target_link_libraries(catboost-metal-train_lib.global PUBLIC ${METAL_TRAIN_LIB_DEPENDENCIES})
target_sources(catboost-metal-train_lib.global PRIVATE
  train.cpp
  trainer_env.cpp
  progress.cpp
  categorical.cpp
  estimated_features.cpp
  estimated_features_apply.cpp
  tree_ctrs.cpp
  ${PROJECT_SOURCE_DIR}/catboost/cuda/ctrs/prior_estimator.cpp
  ${PROJECT_SOURCE_DIR}/contrib/libs/gamma_function_apache_math_port/gamma_function.cpp
  ${PROJECT_SOURCE_DIR}/catboost/libs/model/metal/evaluator.cpp
)
