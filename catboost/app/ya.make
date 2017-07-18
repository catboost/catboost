PROGRAM(catboost)



SRCS(
    cmd_line.cpp
    main.cpp
    mode_calc.cpp
    mode_fit.cpp
    mode_fstr.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/logging
    library/getopt
    library/grid_creator
)

ALLOCATOR(LF)

END()
