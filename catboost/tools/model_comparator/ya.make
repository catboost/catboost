PROGRAM()



PEERDIR(
    catboost/libs/model
    contrib/libs/onnx/proto
    contrib/libs/protobuf
    library/getopt/small
)

SRCS(
    main.cpp
    onnx.cpp
)

END()
