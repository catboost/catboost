PROGRAM(catboost)

DISABLE(USE_ASMLIB)



SRCS(
    main.cpp
    mode_calc.cpp
    mode_fit.cpp
    mode_fstr.cpp
    mode_metadata.cpp
    mode_ostr.cpp
    mode_eval_metrics.cpp
    bind_options.cpp
    cmd_line.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/train_lib
    catboost/libs/data
    catboost/libs/eval_result
    catboost/libs/fstr
    catboost/libs/documents_importance
    catboost/libs/helpers
    catboost/libs/init
    catboost/libs/labels
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/options
    library/getopt/small
    library/grid_creator
    library/json
    library/svnversion
    library/threading/local_executor
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
