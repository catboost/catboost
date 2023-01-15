LIBRARY()

GENERATE_ENUM_SERIALIZATION(node.h)

PEERDIR(
    library/cpp/yson
)



SRCS(
    node.cpp
    node_io.cpp
    node_builder.cpp
    node_visitor.cpp
    serialize.cpp
)

END()

RECURSE_FOR_TESTS(ut)
