LIBRARY()



INCLUDE(${ARCADIA_ROOT}/library/cpp/cuda/wrappers/default_nvcc_flags.make.inc)

SRCS(
    evaluator.cu
    GLOBAL evaluator.cpp
)

PEERDIR(
    catboost/libs/model
    library/cpp/cuda/wrappers
)

END()
