

UNITTEST()
SIZE(MEDIUM)

SRCS(
    calcer_canonization_ut.cpp
    feature_calcer_ut.cpp
    text_processing_collection_ut.cpp
)

PEERDIR(
    catboost/libs/text_features
    catboost/libs/text_features/ut/lib
)


END()
