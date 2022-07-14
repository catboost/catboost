LIBRARY()



SRCS(
    bool.cpp
)

PEERDIR(
    library/cpp/deprecated/atomic
)

END()

RECURSE_FOR_TESTS(
    ut
)
