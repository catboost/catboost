LIBRARY()



SRCS(
    feature_str.cpp
    doc_fstr.cpp
    calc_fstr.cpp
    shap_values.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/model
    library/containers/2d_array
)

END()
