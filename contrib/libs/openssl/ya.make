LIBRARY()

LICENSE(
    OpenSSL
    SSLeay
)



NO_UTIL()

IF (USE_OPENSSL_102)
    PEERDIR(contrib/libs/openssl/1.0.2)
ELSE()
    PEERDIR(contrib/libs/openssl/1.1.1)
ENDIF()

END()
