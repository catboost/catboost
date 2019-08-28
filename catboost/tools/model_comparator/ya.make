PROGRAM()



CFLAGS(-DONNX_ML=1 -DONNX_NAMESPACE=onnx)

PEERDIR(
    catboost/libs/model
    catboost/libs/model/model_export
    contrib/libs/onnx
    contrib/libs/protobuf
    contrib/libs/pugixml
    library/getopt/small
    library/string_utils/ztstrbuf
)

SRCS(
    decl.cpp
    main.cpp
    onnx.cpp
    pmml.cpp
)

END()
