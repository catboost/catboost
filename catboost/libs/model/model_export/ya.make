LIBRARY()



SRCS(
    cpp_exporter.cpp
    export_helpers.cpp
    model_exporter.cpp
    python_exporter.cpp
)

PEERDIR(
    catboost/libs/model/flatbuffers
)

END()
