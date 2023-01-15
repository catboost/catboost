PROTO_LIBRARY()



LICENSE(APACHE2)

SRCS(
    event.proto
    resource_handle.proto
    summary.proto
    tensor.proto
    tensor_shape.proto
    types.proto
)

EXCLUDE_TAGS(JAVA_PROTO)

END()
