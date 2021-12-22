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
    contrib/libs/protobuf/src
)

SRCDIR(contrib/libs/protobuf/src)

SRCS(
    google/protobuf/any.proto
    google/protobuf/api.proto
    google/protobuf/descriptor.proto
    google/protobuf/duration.proto
    google/protobuf/empty.proto
    google/protobuf/field_mask.proto
    google/protobuf/source_context.proto
    google/protobuf/struct.proto
    google/protobuf/timestamp.proto
    google/protobuf/type.proto
    google/protobuf/wrappers.proto
)

END()
