LIBRARY()



SRCS(
    pair.cpp
    query.cpp
    text.cpp
)

PEERDIR(
    catboost/private/libs/index_range
    library/cpp/binsaver
    library/cpp/containers/dense_hash
)

END()
