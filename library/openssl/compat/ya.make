LIBRARY()



PEERDIR(
    contrib/libs/openssl
)

SRCS(
    asn1.c
    bio.c
    bn.c
    crypto.c
    dsa.c
    ecdsa.c
    evp.c
    rsa.c
    ssl.c
    x509.c
    x509_vfy.c
)

END()
