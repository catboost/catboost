LIBRARY()



SRCS(
    analytical_mode_params.cpp
    binarization_options.cpp
    overfitting_detector_options.cpp
    bootstrap_options.cpp
    boosting_options.cpp
    cat_feature_options.cpp
    catboost_options.cpp
    plain_options_helper.cpp
    data_processing_options.cpp
    system_options.cpp
    loss_description.cpp
    metric_options.cpp
    multiclass_label_options.cpp
    oblivious_tree_options.cpp
    output_file_options.cpp
    enum_helpers.cpp
    json_helper.cpp
    check_train_options.cpp
    defaults_helper.cpp
)

PEERDIR(
    library/json
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/ctr_description
    catboost/libs/data_util
    library/getopt/small
    library/grid_creator
    library/json
)

GENERATE_ENUM_SERIALIZATION(enums.h)
GENERATE_ENUM_SERIALIZATION(json_helper.h)

END()
