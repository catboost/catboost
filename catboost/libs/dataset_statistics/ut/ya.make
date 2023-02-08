OWNER(
    eermishkina
    g:matrixnet
)

UNITTEST()

SIZE(MEDIUM)

PEERDIR(
    catboost/libs/dataset_statistics
)

SRCS(
    histograms_ut.cpp
    statistics_serialization_ut.cpp
)

END()
