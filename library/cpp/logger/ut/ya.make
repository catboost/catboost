UNITTEST()



PEERDIR(
    ADDINCL library/cpp/logger
    library/cpp/logger/init_context
    library/cpp/yconf/patcher
)

SRCDIR(library/cpp/logger)

SRCS(
    log_ut.cpp
    element_ut.cpp
    rotating_file_ut.cpp
    composite_ut.cpp
)

END()
