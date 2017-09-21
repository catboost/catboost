PROGRAM()



SRCS(
    pool_converter.cpp
    main.cpp
)

PEERDIR(
    library/getopt
    catboost/cuda/data
)

ALLOCATOR(LF)

END()
