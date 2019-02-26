LIBRARY()



PEERDIR(
    contrib/libs/openssl
    library/openssl/compat # Remove after OpenSSL upgrade
)

SRCS(
    bio.cpp
    x509_vfy.cpp
)

END()

NEED_CHECK()
