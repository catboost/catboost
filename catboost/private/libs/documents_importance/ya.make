LIBRARY()



SRCS(
    docs_importance_helpers.cpp
    docs_importance.cpp
    tree_statistics.cpp
    ders_helpers.cpp
)

PEERDIR(
    catboost/private/libs/algo
    catboost/private/libs/algo_helpers
    catboost/libs/data
    catboost/libs/model
    catboost/private/libs/options
    catboost/libs/helpers
    catboost/private/libs/target
    library/fast_exp
    library/cpp/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(
    enums.h
)

END()
