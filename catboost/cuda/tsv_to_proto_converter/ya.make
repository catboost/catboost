PROGRAM()



SRCS(
    pool_converter.cpp
    main.cpp
)

PEERDIR(
    library/getopt
    catboost/cuda/data
    catboost/libs/column_description
)

ALLOCATOR(LF)

END()
