LIBRARY()



SRCS(
    docs_importance_helpers.cpp
    docs_importance.cpp
    tree_statistics.cpp
    ders_helpers.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/model
    catboost/libs/options
    catboost/libs/metrics
    catboost/libs/helpers
)

GENERATE_ENUM_SERIALIZATION(
    enums.h
)

END()
