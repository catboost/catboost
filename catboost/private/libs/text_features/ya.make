LIBRARY()




SRCS(
    bm25.cpp
    bow.cpp
    embedding_online_features.cpp
    feature_calcer.cpp
    naive_bayesian.cpp
    text_feature_calcers.cpp
    text_processing_collection.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/private/libs/data_types
    catboost/private/libs/options
    catboost/private/libs/text_features/flatbuffers
    catboost/private/libs/text_processing
    contrib/libs/clapack
    contrib/libs/flatbuffers
    library/threading/local_executor
)


END()
