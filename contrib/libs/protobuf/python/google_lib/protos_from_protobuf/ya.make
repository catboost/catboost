PY23_LIBRARY()

LICENSE(BSD-3-Clause)



PY_NAMESPACE(.)
PROTO_NAMESPACE(contrib/libs/protobuf/src)
SRCDIR(contrib/libs/protobuf/src)
PY_SRCS(
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
