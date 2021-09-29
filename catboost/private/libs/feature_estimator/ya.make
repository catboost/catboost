LIBRARY()




SRCS(
    base_embedding_feature_estimator.cpp
    base_text_feature_estimator.cpp
    classification_target.cpp
    embedding_feature_estimators.cpp
    feature_estimator.cpp
    text_feature_estimators.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/private/libs/embeddings
    catboost/private/libs/embedding_features
    catboost/private/libs/text_features
    catboost/private/libs/text_processing
    catboost/private/libs/options
    library/cpp/threading/local_executor
)


END()
