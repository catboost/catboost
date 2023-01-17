LIBRARY()



SRCS(
    dataset_statistics_data_provider_builder.cpp
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
