TOOL()



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/libs/protobuf
    contrib/libs/protobuf/protoc
)

SRCDIR(contrib/libs/grpc-java/compiler/src/java_plugin/cpp)

SRCS(
    java_plugin.cpp
    java_generator.cpp
)

END()
