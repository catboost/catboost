LIBRARY()



SRCS(
    eval_helpers.cpp
)

PEERDIR(
    library/fast_exp
    library/threading/local_executor
    catboost/libs/column_description
    catboost/libs/data
    catboost/libs/data_util
    catboost/libs/helpers
    catboost/libs/labels
    catboost/libs/logging
    catboost/libs/options
)

GENERATE_ENUM_SERIALIZATION(eval_helpers.h)

END()
