PROGRAM()



CFLAGS(-DONNX_ML=1 -DONNX_NAMESPACE=onnx)

PEERDIR(
    catboost/libs/model
    catboost/libs/data_new
    catboost/libs/algo
    contrib/libs/onnx
    contrib/libs/protobuf
    library/getopt/small
)

SRCS(
    main.cpp
    onnx.cpp
)

END()
