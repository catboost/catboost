PROGRAM(catboost)

DISABLE(USE_ASMLIB)



SRCS(
    main.cpp
    mode_calc.cpp
    mode_dataset_statistics.cpp
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
    mode_select_features.cpp
    mode_dump_options.cpp
    GLOBAL signal_handling.cpp
)

PEERDIR(
    catboost/libs/data
    catboost/libs/dataset_statistics
    catboost/libs/features_selection
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/metrics
    catboost/libs/model
    catboost/libs/model/model_export
    catboost/libs/train_lib
    catboost/private/libs/algo
    catboost/private/libs/app_helpers
    catboost/private/libs/data_util
    catboost/private/libs/distributed
    catboost/private/libs/documents_importance
    catboost/private/libs/init
    catboost/private/libs/labels
    catboost/private/libs/options
    catboost/private/libs/target
    library/cpp/getopt/small
    library/cpp/json
    library/cpp/svnversion
    library/cpp/threading/local_executor
)

GENERATE_ENUM_SERIALIZATION(model_metainfo_helpers.h)

IF(OPENSOURCE)
    RESTRICT_LICENSES(
        DENY REQUIRE_DISCLOSURE FORBIDDEN PROTESTWARE
        EXCEPT
            contrib/libs/linux-headers # DTCC-725
            contrib/libs/intel/mkl # DTCC-730
    )
ELSE()
    PEERDIR(
        catboost//private/libs/for_app
    )
ENDIF()

IF (OS_WINDOWS)
    ALLOCATOR(J)
ELSE()
    ALLOCATOR(MIM)
ENDIF()

IF (OPENSOURCE AND AUTOCHECK)
    INCLUDE(${ARCADIA_ROOT}/catboost//oss/checks/check_deps.inc)
ENDIF()

IF (HAVE_CUDA)
    CFLAGS(-DHAVE_CUDA)

    PEERDIR(
        catboost/cuda/cuda_lib
    )
ENDIF()

END()
