LIBRARY()



SRCS(
    embedding_calcers.cpp
    embedding_feature_calcer.cpp
    embedding_processing_collection.cpp
    flatbuffers/embedding_feature_calcers.fbs
    flatbuffers/embedding_processing_collection.fbs
    GLOBAL knn.cpp
    GLOBAL lda.cpp
)

PEERDIR(
    catboost/private/libs/text_features
    catboost/private/libs/text_processing
    contrib/libs/clapack
    contrib/libs/flatbuffers
    library/cpp/hnsw/index
    library/cpp/online_hnsw/dense_vectors
)

END()
