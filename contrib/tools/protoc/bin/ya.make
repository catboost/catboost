PROGRAM(protoc)



LICENSE(BSD-3-Clause)

NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/libs/protoc
)
SRCDIR(
    contrib/libs/protoc
)

SRCS(
    src/google/protobuf/compiler/main.cc
)

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/protoc/ya.make.induced_deps)

END()
