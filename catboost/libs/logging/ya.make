LIBRARY()



SRCS(
    logging.cpp
)

PEERDIR(
    library/cpp/logger
    library/cpp/logger/global
)

GENERATE_ENUM_SERIALIZATION(logging_level.h)

END()
