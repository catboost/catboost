UNITTEST()



PEERDIR(
    ADDINCL library/cpp/logger
)

SRCDIR(library/cpp/logger)

SRCS(
    log_ut.cpp
    element_ut.cpp
    rotating_file_ut.cpp
)

END()
