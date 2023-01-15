

UNITTEST()
SIZE(MEDIUM)

SRCS(
    feature_estimator_ut.cpp
    test_load_embedding.cpp
)

PEERDIR(
    catboost/private/libs/feature_estimator
    catboost/private/libs/text_features
    catboost/private/libs/text_processing
)


END()
