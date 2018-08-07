UNITTEST()



PEERDIR(
    ADDINCL library/logger
)

SRCDIR(library/logger)

SRCS(
    log_ut.cpp
    element_ut.cpp
    rotating_file_ut.cpp
)

END()

NEED_CHECK()
