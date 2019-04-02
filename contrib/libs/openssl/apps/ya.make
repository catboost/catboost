UNION()



IF (USE_OPENSSL_102)
    BUNDLE(contrib/libs/openssl/1.0.2/apps NAME openssl)
ELSE()
    BUNDLE(contrib/libs/openssl/1.1.1/apps NAME openssl)
ENDIF()

END()
