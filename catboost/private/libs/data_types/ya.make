LIBRARY()



SRCS(
    pair.cpp
    query.cpp
    text.cpp
)

PEERDIR(
    catboost/private/libs/index_range
    library/binsaver
    library/containers/dense_hash
)

END()
