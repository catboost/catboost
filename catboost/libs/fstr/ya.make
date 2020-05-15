LIBRARY()



SRCS(
    calc_fstr.cpp
    compare_documents.cpp
    feature_str.cpp
    output_fstr.cpp
    partial_dependence.cpp
    shap_values.cpp
    shap_exact.cpp
    shap_interaction_values.cpp
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
    library/cpp/threading/local_executor
)

END()
