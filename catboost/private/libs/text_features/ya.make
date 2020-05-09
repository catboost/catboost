LIBRARY()




SRCS(
    flatbuffers/feature_calcers.fbs
    flatbuffers/text_processing_collection.fbs
    GLOBAL bm25.cpp
    GLOBAL bow.cpp
    embedding_online_features.cpp
    feature_calcer.cpp
    GLOBAL naive_bayesian.cpp
    text_feature_calcers.cpp
    text_processing_collection.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/private/libs/data_types
    catboost/private/libs/options
    catboost/private/libs/text_processing
    contrib/libs/clapack
    contrib/libs/flatbuffers
    library/cpp/threading/local_executor
)


END()
