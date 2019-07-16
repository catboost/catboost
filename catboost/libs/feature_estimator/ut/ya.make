

UNITTEST()
SIZE(MEDIUM)

SRCS(
    feature_estimator_ut.cpp
    test_load_embedding.cpp
)

PEERDIR(
    catboost/libs/feature_estimator
    catboost/libs/text_features
    catboost/libs/text_processing
)


END()
