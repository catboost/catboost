

UNITTEST()
SIZE(MEDIUM)

SRCS(
    dictionary_ut.cpp
    text_dataset_ut.cpp
)

PEERDIR(
    catboost/private/libs/text_processing
    catboost/private/libs/text_processing/ut/lib
)


END()
