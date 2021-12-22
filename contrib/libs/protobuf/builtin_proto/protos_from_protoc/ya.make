PROTO_LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(BSD-3-Clause)



EXCLUDE_TAGS(
    CPP_PROTO
    GO_PROTO
)

NO_MYPY()

NO_OPTIMIZE_PY_PROTOS()

DISABLE(NEED_GOOGLE_PROTO_PEERDIRS)

PY_NAMESPACE(.)

PROTO_NAMESPACE(
    GLOBAL
    contrib/libs/protoc/src
)

SRCDIR(contrib/libs/protoc/src)

PEERDIR(
    contrib/libs/protobuf/builtin_proto/protos_from_protobuf
)

SRCS(
    google/protobuf/compiler/plugin.proto
)

END()
