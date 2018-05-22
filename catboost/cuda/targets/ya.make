LIBRARY(targets)

NO_WERROR()



SRCS(
    kernel/pointwise_targets.cu
    kernel/query_rmse.cu
    kernel/query_softmax.cu
    kernel/pair_logit.cu
    kernel/yeti_rank_pointwise.cu
    kernel/pfound_f.cu
    kernel/query_cross_entropy.cu
    mse.cpp
    qrmse.cpp
    qsoftmax.cpp
    cross_entropy.cpp
    target_func.cpp
    yeti_rank.cpp
    query_cross_entropy.cpp
    GLOBAL kernel.cpp
    GLOBAL query_cross_entropy_kernels.cpp
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
     -gencode arch=compute_60,code=compute_60
     -gencode arch=compute_61,code=compute_61
     -gencode arch=compute_61,code=sm_61
     -gencode arch=compute_70,code=sm_70
     -gencode arch=compute_70,code=compute_70
)



END()
