

LIBRARY()

PEERDIR(
    catboost/libs/algo
    catboost/libs/data_new
    library/getopt/small
    library/threading/future
)

SRCS(
    GLOBAL main.cpp
    GLOBAL perftest_modules.cpp
)

END()
