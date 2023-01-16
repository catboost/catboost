PY23_LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(BSD-3-Clause)



PEERDIR(
    contrib/libs/protobuf
)

NO_MYPY()

PY_NAMESPACE(.)

PROTO_NAMESPACE(contrib/libs/protoc/src)

SRCDIR(contrib/libs/protoc/src)

PY_SRCS(google/protobuf/compiler/plugin.proto)

END()
