LIBRARY()



SRCS(
    features_layout.cpp
    meta_info.cpp
)

PEERDIR(
    catboost/libs/column_description
    catboost/libs/data_types
    catboost/libs/helpers
    catboost/libs/model
    catboost/libs/options
)

END()
