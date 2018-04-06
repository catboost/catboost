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


CUDA_NVCC_FLAGS(
    -gencode arch=compute_30,code=compute_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=compute_61
)

ALLOCATOR(LF)


END()
