PROTO_LIBRARY()



LICENSE(MIT)

CFLAGS(-DONNX_ML=1 -DONNX_NAMESPACE=onnx)

SRCS(
    onnx_ml.proto
    onnx_operators_ml.proto
)

EXCLUDE_TAGS(JAVA_PROTO)  # TODO: remove (DEVTOOLS-3496)

END()
