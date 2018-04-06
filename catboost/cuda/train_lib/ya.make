LIBRARY()




SRCS(
    GLOBAL train.cpp
    GLOBAL cross_entropy.cpp
    GLOBAL pair_logit.cpp
    GLOBAL yeti_rank.cpp
    GLOBAL pointwise.cpp
    GLOBAL qrmse.cpp
    GLOBAL qsoftmax.cpp
    GLOBAL rmse.cpp
)

PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/data
    catboost/cuda/ctrs
    catboost/cuda/gpu_data
    catboost/cuda/methods
    catboost/cuda/models
    catboost/cuda/targets
    catboost/cuda/cpu_compatibility_helpers
    catboost/libs/model
    catboost/libs/logging
    catboost/libs/data
    catboost/libs/algo
    catboost/libs/train_lib
    catboost/libs/helpers
    library/grid_creator
)


CUDA_NVCC_FLAGS(
    --expt-relaxed-constexpr
     -gencode arch=compute_30,code=compute_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_61,code=compute_61
)

END()
