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
    rsa.c
    ssl.c
    x509.c
)

END()
