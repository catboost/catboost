LIBRARY(targets)

NO_WERROR()



SRCS(
    kernel/pointwise_targets.cu
    mse.cpp
    cross_entropy.cpp
    target_base.cpp
    quality_metric_helpers.cpp
)


PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/gpu_data
)

CUDA_NVCC_FLAGS(
    --expt-relaxed-constexpr
    -std=c++11
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=compute_61
    --ptxas-options=-v
)


GENERATE_ENUM_SERIALIZATION(target_options.h)

END()
