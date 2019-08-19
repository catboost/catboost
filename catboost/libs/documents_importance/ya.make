LIBRARY()



SRCS(
    docs_importance_helpers.cpp
    docs_importance.cpp
    tree_statistics.cpp
    ders_helpers.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/algo_helpers
    catboost/libs/data_new
    catboost/libs/model
    catboost/libs/options
    catboost/libs/helpers
    catboost/libs/target
    library/fast_exp
    library/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(
    enums.h
)

END()
