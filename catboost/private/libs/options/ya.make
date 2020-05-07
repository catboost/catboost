

LIBRARY()

SRCS(
    analytical_mode_params.cpp
    binarization_options.cpp
    boosting_options.cpp
    bootstrap_options.cpp
    cat_feature_options.cpp
    catboost_options.cpp
    check_train_options.cpp
    class_label_options.cpp
    cross_validation_params.cpp
    data_processing_options.cpp
    dataset_reading_params.cpp
    defaults_helper.cpp
    enum_helpers.cpp
    feature_eval_options.cpp
    feature_penalties_options.cpp
    json_helper.cpp
    load_options.cpp
    loss_description.cpp
    metric_options.cpp
    model_based_eval_options.cpp
    monotone_constraints.cpp
    oblivious_tree_options.cpp
    option.cpp
    output_file_options.cpp
    overfitting_detector_options.cpp
    parse_per_feature_options.cpp
    plain_options_helper.cpp
    runtime_text_options.cpp
    split_params.cpp
    system_options.cpp
    text_processing_options.cpp
)

PEERDIR(
    library/json
    catboost/libs/helpers
    catboost/libs/logging
    catboost/private/libs/ctr_description
    catboost/private/libs/data_util
    library/getopt/small
    library/cpp/grid_creator
    library/json
    library/text_processing/dictionary
    library/text_processing/tokenizer
)

GENERATE_ENUM_SERIALIZATION(enums.h)
GENERATE_ENUM_SERIALIZATION(json_helper.h)

END()
