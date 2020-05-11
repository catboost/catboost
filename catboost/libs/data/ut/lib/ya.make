LIBRARY()



SRCS(
    for_data_provider.cpp
    for_loader.cpp
    for_objects.cpp
    for_target.cpp
)

PEERDIR(
    library/cpp/unittest

    catboost/libs/cat_feature
    catboost/libs/data
    catboost/private/libs/data_types
    catboost/private/libs/data_util
    catboost/libs/helpers
    catboost/private/libs/options
)

END()
