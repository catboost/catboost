LIBRARY()



SRCS(
    feature_str.cpp
    calc_fstr.cpp
    output_fstr.cpp
    shap_values.cpp
    util.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/data_new
    catboost/libs/helpers
    catboost/libs/loggers
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/options
    library/threading/local_executor
)

END()
