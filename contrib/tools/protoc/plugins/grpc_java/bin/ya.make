

PROGRAM(grpc_java)

INCLUDE(ya.version)

VERSION(${GRPC_JAVA_VERSION})

NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/libs/protoc
)

SRCDIR(contrib/libs/grpc-java/compiler/src/java_plugin/cpp)

SRCS(
    java_plugin.cpp
    java_generator.cpp
)

END()
