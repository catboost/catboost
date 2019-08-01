LIBRARY()



SRCS(
    calc_fstr.cpp
    compare_documents.cpp
    feature_str.cpp
    output_fstr.cpp
    shap_values.cpp
    util.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data_new
    catboost/libs/helpers
    catboost/libs/loggers
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/options
    catboost/libs/target
    library/threading/local_executor
)

END()
