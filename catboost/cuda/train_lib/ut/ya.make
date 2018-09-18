

UNITTEST_FOR(catboost/cuda/train_lib)

SRCS(
    dummy_ut.cpp
)

IF (NOT AUTOCHECK)
    SRCS(
        train_ut.cpp
    )
ENDIF()

PEERDIR(
    catboost/libs/train_lib
    catboost/libs/model
)

END()
