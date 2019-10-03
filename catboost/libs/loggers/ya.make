RECURSE(
    tensorboard_logger_example
)

LIBRARY()



SRCS(
    tensorboard_logger.cpp
    catboost_logger_helpers.cpp
    logger.cpp
)

PEERDIR(
    catboost/libs/logging
    catboost/private/libs/options
    catboost/libs/metrics
    contrib/libs/tensorboard
    library/digest/crc32c
    library/json/writer
)

END()
