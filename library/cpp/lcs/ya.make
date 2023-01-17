LIBRARY()



PEERDIR(
    library/cpp/containers/paged_vector
)

SRCS(
    lcs_via_lis.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
