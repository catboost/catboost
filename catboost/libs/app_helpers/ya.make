LIBRARY()



SRCS(
    mode_calc_helpers.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/eval_result
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/options
    library/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(implementation_type_enum.h)

END()
