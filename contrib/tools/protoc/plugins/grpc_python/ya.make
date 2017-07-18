TOOL()



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/libs/protobuf
    contrib/libs/protobuf/protoc
)

ADDINCL(
    contrib/libs/grpc
    contrib/libs/grpc/include
)

SRCDIR(contrib/libs/grpc/src/compiler)

SRCS(
    python_plugin.cc
    python_generator.cc
)

END()
