LIBRARY()



PEERDIR(
    contrib/libs/openssl
)

SRCS(
    bio.cpp
    x509_vfy.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
