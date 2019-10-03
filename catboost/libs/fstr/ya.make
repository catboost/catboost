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
    catboost/private/libs/algo
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/loggers
    catboost/libs/logging
    catboost/libs/model
    catboost/private/libs/options
    catboost/private/libs/target
    library/threading/local_executor
)

END()
