LIBRARY()



SRCS(
    approx_calcer_helpers.cpp
    approx_updater_helpers.cpp

    custom_objective_descriptor.cpp
    ders_holder.cpp
    error_functions.cpp
    hessian.cpp
    online_predictor.cpp
    pairwise_leaves_calculation.cpp
)

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/model
    catboost/libs/options
)

END()
