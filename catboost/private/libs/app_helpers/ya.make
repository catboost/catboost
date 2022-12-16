LIBRARY()



SRCS(
    bind_options.cpp
    mode_calc_helpers.cpp
    mode_dataset_statistics_helpers.cpp
    mode_fit_helpers.cpp
    mode_fstr_helpers.cpp
    mode_normalize_model_helpers.cpp
)

PEERDIR(
    catboost/private/libs/algo
    catboost/libs/column_description
    catboost/libs/data
    catboost/libs/dataset_statistics
    catboost/libs/eval_result
    catboost/libs/fstr
    catboost/libs/helpers
    catboost/private/libs/index_range
    catboost/private/libs/labels
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/train_lib
    catboost/private/libs/options
    library/cpp/getopt/small
    library/cpp/grid_creator
    library/cpp/json
    library/cpp/logger
    library/cpp/object_factory
    library/cpp/text_processing/dictionary
    library/cpp/threading/local_executor
)

IF(HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
        catboost/libs/model/cuda
    )
ENDIF()

GENERATE_ENUM_SERIALIZATION(implementation_type_enum.h)

END()
