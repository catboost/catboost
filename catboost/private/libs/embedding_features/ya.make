LIBRARY()



SRCS(
    lda.cpp
    embedding_calcers.cpp
    embedding_feature_calcer.cpp
)

PEERDIR(
    catboost/private/libs/text_features
    catboost/private/libs/text_processing
    contrib/libs/clapack
    contrib/libs/flatbuffers
)

END()
