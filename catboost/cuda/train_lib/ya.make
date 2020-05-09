LIBRARY()




SRCS(
    GLOBAL train.cpp
    GLOBAL query_cross_entropy.cpp
    GLOBAL pointwise.cpp
    GLOBAL pointwise_region.cpp
    GLOBAL querywise.cpp
    GLOBAL pfound_f.cpp
    GLOBAL pair_logit_pairwise.cpp
    GLOBAL multiclass.cpp
    GLOBAL multiclass_region.cpp
    GLOBAL querywise_region.cpp
    GLOBAL pointwise_non_symmetric.cpp
    GLOBAL querywise_non_symmetric.cpp
    GLOBAL multiclass_non_symmetric.cpp
    GLOBAL combination.cpp
)

PEERDIR(
    catboost/cuda/cpu_compatibility_helpers
    catboost/cuda/ctrs
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/data
    catboost/cuda/gpu_data
    catboost/cuda/methods
    catboost/cuda/models
    catboost/cuda/targets
    catboost/private/libs/algo
    catboost/private/libs/algo_helpers
    catboost/libs/data
    catboost/libs/eval_result
    catboost/libs/fstr
    catboost/libs/helpers
    catboost/libs/loggers
    catboost/libs/model
    catboost/private/libs/options
    catboost/libs/overfitting_detector
    catboost/private/libs/quantization
    catboost/libs/train_lib
    library/cpp/json
    library/object_factory
    library/cpp/threading/local_executor
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
