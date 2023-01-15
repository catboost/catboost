LIBRARY()



SRCS(
    embedding_calcers.cpp
    embedding_feature_calcer.cpp
    knn.cpp
    lda.cpp
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
