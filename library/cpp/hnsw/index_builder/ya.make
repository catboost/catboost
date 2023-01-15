LIBRARY()



SRCS(
    index_writer.cpp
    build_routines.cpp
)

PEERDIR(
    library/cpp/dot_product
    library/cpp/hnsw/helpers
    library/cpp/hnsw/logging
    library/cpp/containers/dense_hash
    library/cpp/threading/local_executor
    library/cpp/json
    util
)

END()
