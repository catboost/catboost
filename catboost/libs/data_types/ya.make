LIBRARY()



SRCS(
    pair.cpp
    text.cpp
)

PEERDIR(
    catboost/libs/index_range
    library/binsaver
    library/containers/dense_hash
)

END()
