

LIBRARY()

PEERDIR(
    catboost/private/libs/algo
    catboost/libs/data
    library/getopt/small
    library/threading/future
)

SRCS(
    GLOBAL main.cpp
    GLOBAL perftest_modules.cpp
)

END()
