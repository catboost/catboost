

PROGRAM(grpc_cpp)

LICENSE(Apache-2.0)

PEERDIR(
    contrib/libs/grpc/src/compiler/grpc_plugin_support
)

ADDINCL(
    contrib/libs/grpc
    contrib/libs/grpc/include
)

NO_COMPILER_WARNINGS()

CFLAGS(
    -DGRPC_USE_ABSL=0
)

SRCDIR(contrib/libs/grpc/src/compiler)

SRCS(
    cpp_plugin.cc
)

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/protoc/plugins/grpc_cpp/ya.make.induced_deps)

END()
