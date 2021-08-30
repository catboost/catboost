LIBRARY()



SRCS(
    all.cpp
    index_reader.cpp
)

PEERDIR(
    library/cpp/containers/dense_hash
    library/cpp/dot_product
    library/cpp/l1_distance
    library/cpp/l2_distance
    util
)

END()
