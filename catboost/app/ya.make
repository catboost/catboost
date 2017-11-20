PROGRAM(catboost)

DISABLE(USE_ASMLIB)



SRCS(
    cmd_line.cpp
    main.cpp
    mode_calc.cpp
    mode_fit.cpp
    mode_fstr.cpp
    mode_plot.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/fstr
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/params
    library/getopt/small
    library/grid_creator
    library/json
    library/malloc/api
    library/svnversion
    library/threading/local_executor
)

NO_GPL()

ALLOCATOR(LF)

END()
