LIBRARY()

LICENSE(
    OpenSSL
    SSLeay
)



NO_UTIL()

IF (USE_OPENSSL_111)
    PEERDIR(contrib/libs/openssl/1.1.1)
ELSE()
    PEERDIR(contrib/libs/openssl/1.0.2)
ENDIF()

END()
