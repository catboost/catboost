LIBRARY()

NO_WERROR()



SRCS(
    kernel/ctr_calcers.cu
    ctr_bins_builder.cpp
    ctr_calcers.cpp
    GLOBAL ctr_kernels.cpp
    ctr.cpp
)

PEERDIR(
    build/cuda
    catboost/cuda/cuda_lib
    catboost/libs/ctr_description
    catboost/cuda/cuda_util
    catboost/cuda/utils
    contrib/libs/gamma_function_apache_math_port
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

GENERATE_ENUM_SERIALIZATION(ctr.h)


END()
