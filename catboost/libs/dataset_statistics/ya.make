LIBRARY()



SRCS(
    dataset_statistics_data_provider_builder.cpp
    statistics_data_structures.cpp
)

PEERDIR(
    catboost/libs/data
    catboost/private/libs/options
    catboost/private/libs/target
    library/cpp/json
)

END()
