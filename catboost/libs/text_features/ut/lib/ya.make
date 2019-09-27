LIBRARY()



SRCS(
    text_features_data.cpp
)

PEERDIR(
    catboost/libs/options
    catboost/libs/text_features
    catboost/libs/text_processing
    catboost/libs/text_processing/ut/lib
)


END()
