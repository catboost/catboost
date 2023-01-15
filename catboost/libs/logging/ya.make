LIBRARY()



SRCS(
    logging.cpp
)

PEERDIR(
    library/logger
    library/logger/global
)

GENERATE_ENUM_SERIALIZATION(logging_level.h)

END()
