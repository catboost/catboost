LIBRARY()



SRCS(
    kernel/ctr_calcers.cu
    ctr_bins_builder.cpp
    ctr_calcers.cpp
    GLOBAL ctr_kernels.cpp
    ctr.cpp
    prior_estimator.cpp
)

PEERDIR(
    build/platform/cuda
    catboost/cuda/cuda_lib
    catboost/private/libs/ctr_description
    catboost/cuda/cuda_util
    catboost/libs/helpers
    contrib/libs/gamma_function_apache_math_port
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

GENERATE_ENUM_SERIALIZATION(ctr.h)


END()
