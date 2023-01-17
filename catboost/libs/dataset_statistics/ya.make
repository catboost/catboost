LIBRARY()



SRCS(
    visitors.cpp
    histograms.cpp
    statistics_data_structures.cpp
)

GENERATE_ENUM_SERIALIZATION(histograms.h)

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/data
    catboost/private/libs/options
    catboost/private/libs/target
    library/cpp/json
)

END()
