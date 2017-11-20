LIBRARY()



SRCS(
    feature_str.cpp
    doc_fstr.cpp
    calc_fstr.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/model
    catboost/libs/params
    library/containers/2d_array
)

END()
