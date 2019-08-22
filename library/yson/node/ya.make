LIBRARY()

GENERATE_ENUM_SERIALIZATION(node.h)

PEERDIR(
    library/yson
)



SRCS(
    node.cpp
    node_io.cpp
    node_builder.cpp
    node_visitor.cpp
    serialize.cpp
)

END()
