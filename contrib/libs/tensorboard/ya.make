PROTO_LIBRARY()

LICENSE(Apache-2.0)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

VERSION(0.12.1)

ORIGINAL_SOURCE(https://github.com/tensorflow/tensorflow)



SRCS(
    event.proto
    resource_handle.proto
    summary.proto
    tensor.proto
    tensor_shape.proto
    types.proto
)

# TODO: remove (DEVTOOLS-3496)
EXCLUDE_TAGS(
    GO_PROTO
    JAVA_PROTO
)

END()
