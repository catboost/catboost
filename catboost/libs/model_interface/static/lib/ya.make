

LIBRARY()

SRCDIR(catboost/libs/model_interface)

SRCS(
    c_api.cpp
)

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/model
)

END()

