LIBRARY()




SRCS(
    base_text_feature_estimator.cpp
    feature_estimator.cpp
    text_feature_estimators.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/private/libs/text_features
    catboost/private/libs/text_processing
    catboost/private/libs/options
    library/cpp/threading/local_executor
)


END()
