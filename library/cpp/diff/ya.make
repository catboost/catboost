LIBRARY()



PEERDIR(
    library/cpp/lcs
    library/cpp/containers/stack_array
)

SRCS(
    diff.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
