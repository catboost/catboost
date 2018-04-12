LIBRARY(targets)

NO_WERROR()



SRCS(
    kernel/pointwise_targets.cu
    kernel/query_rmse.cu
    kernel/query_softmax.cu
    kernel/pair_logit.cu
    kernel/yeti_rank_pointwise.cu
    mse.cpp
    qrmse.cpp
    qsoftmax.cpp
    cross_entropy.cpp
    target_func.cpp
    yeti_rank.cpp
    GLOBAL kernel.cpp
)


PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/gpu_data
    catboost/libs/options
    catboost/libs/metrics
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
