

LIBRARY()

SRCS(
    converter.cpp
)

PEERDIR(
    library/cpp/json/writer
)

END()

RECURSE_FOR_TESTS(
    ut
)
