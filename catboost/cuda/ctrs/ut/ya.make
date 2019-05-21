

IF (AUTOCHECK)
    LIBRARY()
ELSE()
    UNITTEST()
ENDIF()

SRCS(
    test_ctrs.cpp
)

PEERDIR(
    catboost/cuda/ctrs
    catboost/libs/helpers
    catboost/cuda/data
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
