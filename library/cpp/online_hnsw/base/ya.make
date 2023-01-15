LIBRARY()



SRCS(
    build_options.cpp
    dynamic_dense_graph.cpp
    index_base.cpp
    index_reader.cpp
    index_writer.cpp
    item_storage_index.cpp
)

PEERDIR(
    library/cpp/hnsw/index_builder
    library/cpp/containers/dense_hash
    library/cpp/threading/local_executor
    util
)

END()
