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
