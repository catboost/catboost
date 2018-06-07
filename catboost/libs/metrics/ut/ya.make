

UNITTEST()

PEERDIR(
    catboost/libs/metrics
)

SRCS(
    balanced_accuracy_ut.cpp
    dcg_ut.cpp
    kappa_ut.cpp
)

END()
