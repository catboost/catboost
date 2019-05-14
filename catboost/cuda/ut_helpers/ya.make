

LIBRARY()

SRCS(
    test_utils.cpp
)

PEERDIR(
    catboost/cuda/data
    catboost/libs/data_new
    catboost/libs/data_types
    catboost/libs/data_util
    catboost/libs/helpers
    catboost/libs/labels
    catboost/libs/options
    catboost/libs/quantization
    catboost/libs/train_lib
)

END()
