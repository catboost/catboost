LIBRARY()



SRCS(
    tensorboard_logger.cpp
)

PEERDIR(
    contrib/libs/tensorboard
    library/digest/crc32c
)

END()
