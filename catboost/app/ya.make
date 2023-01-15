PROGRAM(catboost)

DISABLE(USE_ASMLIB)



SRCS(
    bind_options.cpp
    main.cpp
    mode_calc.cpp
    mode_eval_metrics.cpp
    mode_eval_feature.cpp
    mode_fit.cpp
    mode_fstr.cpp
    mode_metadata.cpp
    mode_model_based_eval.cpp
    mode_model_sum.cpp
    mode_normalize_model.cpp
    mode_ostr.cpp
    mode_roc.cpp
    mode_run_worker.cpp
    GLOBAL signal_handling.cpp
)

PEERDIR(
    catboost/private/libs/algo
    catboost/private/libs/app_helpers
    catboost/libs/column_description
    catboost/libs/data
    catboost/private/libs/data_util
    catboost/private/libs/distributed
    catboost/private/libs/documents_importance
    catboost/libs/helpers
    catboost/private/libs/init
    catboost/private/libs/labels
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/model/model_export
    catboost/private/libs/options
    catboost/private/libs/target
    catboost/libs/train_lib
    library/cpp/getopt/small
    library/cpp/grid_creator
    library/cpp/json
    library/cpp/logger
    library/svnversion
    library/cpp/text_processing/dictionary
    library/cpp/threading/local_executor
)

IF(HAVE_CUDA)
    PEERDIR(
        catboost/cuda/train_lib
        catboost/libs/model/cuda
    )
ENDIF()

GENERATE_ENUM_SERIALIZATION(model_metainfo_helpers.h)

IF(CATBOOST_OPENSOURCE)
    NO_GPL()
ELSE()
    PEERDIR(
        catboost//private/libs/for_app
    )
ENDIF()

IF (ARCH_AARCH64 OR OS_WINDOWS)
    ALLOCATOR(J)
ELSE()
    ALLOCATOR(LF)
ENDIF()

IF (CATBOOST_OPENSOURCE AND AUTOCHECK)
    INCLUDE(${ARCADIA_ROOT}/catboost//oss/checks/check_deps.inc)
ENDIF()

END()
