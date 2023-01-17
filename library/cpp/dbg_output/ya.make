LIBRARY()



PEERDIR(
    library/cpp/colorizer
)

SRCS(
    dump.cpp
    dumpers.cpp
    engine.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
