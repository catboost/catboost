LIBRARY()



SRCS(
    text_features_data.cpp
)

PEERDIR(
    catboost/private/libs/options
    catboost/private/libs/text_features
    catboost/private/libs/text_processing
    catboost/private/libs/text_processing/ut/lib
)


END()
