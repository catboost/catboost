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

SRCDIR(contrib/libs/grpc)

SRCS(
    src/compiler/cpp_plugin.cc
    src/compiler/cpp_generator.cc
)

END()
