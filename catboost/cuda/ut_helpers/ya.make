

LIBRARY()

SRCS(
    test_utils.cpp
)

PEERDIR(
    catboost/cuda/data
    catboost/libs/data
    catboost/private/libs/data_types
    catboost/private/libs/data_util
    catboost/libs/helpers
    catboost/private/libs/labels
    catboost/private/libs/options
    catboost/private/libs/quantization
    catboost/libs/train_lib
)

END()
