LIBRARY()



SRCS(
    label_converter.cpp
    external_label_helper.cpp
    helpers.cpp
)

PEERDIR(
    catboost/libs/logging
    catboost/libs/model
    catboost/private/libs/options
    library/cpp/json
)

END()
