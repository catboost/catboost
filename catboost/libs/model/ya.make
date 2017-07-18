LIBRARY()



SRCS(
    coreml_helpers.cpp
    model.cpp
    model_calcer.cpp
    tensor_struct.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/libs/logging
    library/containers/2d_array
    library/json
    contrib/libs/coreml
)

END()
