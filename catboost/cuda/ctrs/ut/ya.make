UNITTEST(ctrs_ut)

NO_WERROR()



IF (NOT AUTOCHECK)
SRCS(
    test_ctrs.cpp
)
ENDIF()


PEERDIR(
    catboost/cuda/ctrs
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

ALLOCATOR(LF)


END()
