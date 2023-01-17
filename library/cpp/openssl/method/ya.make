LIBRARY()



PEERDIR(
    contrib/libs/openssl
    library/cpp/openssl/holders
)

SRCS(
    io.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
