

IF (AUTOCHECK)
    LIBRARY()
ELSE()
    UNITTEST()
ENDIF()

SRCS(
    test_tree_searcher.cpp
    test_pairwise_tree_searcher.cpp
    test_multistat_histograms.cpp
)

PEERDIR(
    catboost/cuda/gpu_data
    catboost/cuda/data
    catboost/cuda/methods
    catboost/cuda/ut_helpers
    catboost/libs/helpers
    catboost/private/libs/quantization
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
