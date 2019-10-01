LIBRARY()



SRCS(
    for_data_provider.cpp
    for_loader.cpp
    for_objects.cpp
    for_target.cpp
)

PEERDIR(
    library/unittest

    catboost/libs/cat_feature
    catboost/libs/data
    catboost/libs/data_types
    catboost/libs/data_util
    catboost/libs/helpers
    catboost/libs/options
)

END()
