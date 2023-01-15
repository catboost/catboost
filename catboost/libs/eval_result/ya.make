LIBRARY()



SRCS(
    column_printer.cpp
    eval_helpers.cpp
    eval_result.cpp
    pool_printer.cpp
)

PEERDIR(
    library/cpp/threading/local_executor
    catboost/libs/column_description
    catboost/libs/data
    catboost/private/libs/data_util
    catboost/libs/helpers
    catboost/idl/pool/flat
    catboost/private/libs/labels
    catboost/libs/logging
    catboost/libs/model
    catboost/private/libs/options
    catboost/private/libs/quantized_pool
)

GENERATE_ENUM_SERIALIZATION(eval_helpers.h)

END()
