RECURSE(
    tensorboard_logger_example
)

LIBRARY()



SRCS(
    tensorboard_logger.cpp
    logger.cpp
)

PEERDIR(
    catboost/libs/logging
    contrib/libs/tensorboard
    library/digest/crc32c
)

END()
