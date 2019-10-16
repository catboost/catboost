

LIBRARY()

PEERDIR(
    catboost/private/libs/algo
    catboost/libs/data
    library/getopt/small
    library/threading/future
)

IF (HAVE_CUDA)
    PEERDIR(
        catboost/libs/model/cuda
    )
ENDIF()

SRCS(
    GLOBAL main.cpp
    GLOBAL perftest_modules.cpp
)

END()
