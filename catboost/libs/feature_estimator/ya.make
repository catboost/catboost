LIBRARY()




SRCS(
    base_text_feature_estimator.cpp
    feature_estimator.cpp
    text_feature_estimators.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/libs/text_features
    catboost/libs/text_processing
    catboost/libs/options
    library/threading/local_executor
)


END()
