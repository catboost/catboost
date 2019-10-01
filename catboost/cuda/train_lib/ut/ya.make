

IF (AUTOCHECK)
    LIBRARY()
    PEERDIR(catboost/cuda/train_lib)
    ADDINCL(catboost/cuda/train_lib)
ELSE()
    UNITTEST_FOR(catboost/cuda/train_lib)
ENDIF()

SRCS(
    dummy_ut.cpp
    train_ut.cpp
)

PEERDIR(
    catboost/libs/data
    catboost/libs/train_lib
    catboost/libs/model
)

END()
