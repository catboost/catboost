LIBRARY(models)

NO_WERROR()



SRCS(
    kernel/add_model_value.cu
    GLOBAL add_bin_values.cpp
    add_model.cpp
)


PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/gpu_data
)

CUDA_NVCC_FLAGS(
    --expt-relaxed-constexpr
     -gencode arch=compute_30,code=compute_30  -gencode arch=compute_35,code=sm_35  -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_61,code=compute_61
)



END()
