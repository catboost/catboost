

PROGRAM(grpc_python)

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
    python_plugin.cc
)

END()
