PROGRAM()



SRCS(
    main.cpp
)

PEERDIR(
    library/cpp/hnsw/index
    library/cpp/hnsw/index_builder
    library/cpp/getopt/small
    util
)

GENERATE_ENUM_SERIALIZATION(distance.h)

GENERATE_ENUM_SERIALIZATION(vector_component_type.h)

ALLOCATOR(LF)

END()
