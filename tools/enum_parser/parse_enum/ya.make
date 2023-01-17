LIBRARY()



SRCS(
    parse_enum.cpp
)

PEERDIR(
    library/cpp/cppparser
)

END()

RECURSE(
    benchmark
    ut
)

RECURSE_FOR_TESTS(
    benchmark_build
)
