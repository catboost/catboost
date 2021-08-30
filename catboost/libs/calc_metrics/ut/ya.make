

UNITTEST()

SIZE(MEDIUM)

PEERDIR(
    catboost/libs/model/ut/lib
    catboost/libs/calc_metrics
)

SRCS(
    calc_metrics_ut.cpp
)

END()
