PY23_LIBRARY()

LICENSE(BSD-3-Clause)



PEERDIR(contrib/libs/protobuf)

PY_NAMESPACE(.)
PROTO_NAMESPACE(contrib/libs/protoc/src)
SRCDIR(contrib/libs/protoc/src)
PY_SRCS(
    google/protobuf/compiler/plugin.proto
)

END()
