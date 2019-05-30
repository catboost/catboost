LIBRARY()




SRCS(
    embedding.cpp
    embedding_loader.cpp
    embedding_online_features.cpp
    estimators.cpp
    naive_bayesian.cpp
    bm25.cpp
    text_column_builder.cpp
    text_dataset.cpp
    tokenizer.cpp
    bow.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/libs/model
    catboost/libs/data_types
    catboost/libs/feature_estimator
    catboost/libs/options
    contrib/libs/clapack
    library/text_processing/dictionary
    library/threading/local_executor
)


END()
