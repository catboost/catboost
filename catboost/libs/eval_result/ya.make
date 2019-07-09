LIBRARY()



SRCS(
    column_printer.cpp
    eval_helpers.cpp
    eval_result.cpp
    pool_printer.cpp
)

PEERDIR(
    library/threading/local_executor
    catboost/libs/column_description
    catboost/libs/data_new
    catboost/libs/data_util
    catboost/libs/helpers
    catboost/idl/pool/flat
    catboost/libs/labels
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/options
    catboost/libs/quantized_pool
)

GENERATE_ENUM_SERIALIZATION(eval_helpers.h)

END()
