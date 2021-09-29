PROTO_LIBRARY()

LICENSE(MIT)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)



CFLAGS(
    -DONNX_ML=1
    -DONNX_NAMESPACE=onnx
)

SRCS(
    onnx_ml.proto
    onnx_operators_ml.proto
)

# TODO: remove (DEVTOOLS-3496)
EXCLUDE_TAGS(
    GO_PROTO
    JAVA_PROTO
)

END()
