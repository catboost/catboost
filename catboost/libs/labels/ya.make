LIBRARY()



SRCS(
    label_converter.cpp
    external_label_helper.cpp
)

PEERDIR(
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/options
)

END()
