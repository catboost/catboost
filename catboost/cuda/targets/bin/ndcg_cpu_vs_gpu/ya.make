

PROGRAM()

SRCS(
    main.cpp
)

PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/targets
    catboost/libs/logging
    catboost/libs/metrics
    catboost/private/libs/options
    library/getopt
    library/threading/local_executor
)

END()
