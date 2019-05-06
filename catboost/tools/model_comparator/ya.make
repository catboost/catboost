PROGRAM()



CFLAGS(-DONNX_ML=1 -DONNX_NAMESPACE=onnx)

PEERDIR(
    catboost/libs/model
    contrib/libs/onnx
    contrib/libs/protobuf
    library/getopt/small
)

SRCS(
    main.cpp
    onnx.cpp
)

END()
