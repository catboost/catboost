LIBRARY()



SRCS(
    mode_calc_helpers.cpp
    mode_fstr_helpers.cpp
    mode_normalize_model_helpers.cpp
)

PEERDIR(
    catboost/private/libs/algo
    catboost/libs/column_description
    catboost/libs/data
    catboost/libs/eval_result
    catboost/libs/fstr
    catboost/libs/helpers
    catboost/private/libs/labels
    catboost/libs/logging
    catboost/libs/model
    catboost/private/libs/options
    library/getopt/small
    library/object_factory
    library/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(implementation_type_enum.h)

END()
