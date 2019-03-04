PROGRAM(catboost)

DISABLE(USE_ASMLIB)



SRCS(
    bind_options.cpp
    main.cpp
    mode_calc.cpp
    mode_eval_metrics.cpp
    mode_fit.cpp
    mode_fstr.cpp
    mode_metadata.cpp
    mode_model_based_eval.cpp
    mode_model_sum.cpp
    mode_ostr.cpp
    mode_roc.cpp
    mode_run_worker.cpp
    GLOBAL signal_handling.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/app_helpers
    catboost/libs/column_description
    catboost/libs/data_new
    catboost/libs/data_util
    catboost/libs/distributed
    catboost/libs/documents_importance
    catboost/libs/helpers
    catboost/libs/init
    catboost/libs/labels
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/options
    catboost/libs/target
    catboost/libs/train_lib
    library/getopt/small
    library/grid_creator
    library/json
    library/logger
    library/svnversion
)

IF(HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
    )
ENDIF()

GENERATE_ENUM_SERIALIZATION(model_metainfo_helpers.h)

IF(CATBOOST_OPENSOURCE)
    NO_GPL()
ELSE()
    PEERDIR(
        catboost//libs/for_app
    )
ENDIF()

ALLOCATOR(LF)

END()
