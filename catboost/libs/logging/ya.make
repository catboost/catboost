LIBRARY()



PEERDIR(
    library/logger/global
)

SRCS(
    logging.cpp
)

GENERATE_ENUM_SERIALIZATION(logging_level.h)

END()
