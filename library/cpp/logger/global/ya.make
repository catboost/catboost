LIBRARY()



PEERDIR(
    library/cpp/logger
)

SRCS(
    common.cpp
    global.cpp
    rty_formater.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
