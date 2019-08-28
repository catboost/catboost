LIBRARY()



INCLUDE(${ARCADIA_ROOT}/catboost/libs/cuda_wrappers/default_nvcc_flags.make.inc)

SRCS(
    evaluator.cu
    GLOBAL evaluator.cpp
)

PEERDIR(
    catboost/libs/cuda_wrappers
)

PEERDIR(
    catboost/libs/model
)

END()
