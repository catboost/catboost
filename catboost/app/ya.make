PROGRAM(catboost)

DISABLE(USE_ASMLIB)



SRCS(
    main.cpp
    mode_calc.cpp
    mode_fit.cpp
    mode_fstr.cpp
    mode_ostr.cpp
    mode_eval_metrics.cpp
    bind_options.cpp
    cmd_line.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/train_lib
    catboost/libs/data
    catboost/libs/fstr
    catboost/libs/documents_importance
    catboost/libs/helpers
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

IF(CATBOOST_OPENSOURCE)
    NO_GPL()
ENDIF()

ALLOCATOR(LF)

END()
