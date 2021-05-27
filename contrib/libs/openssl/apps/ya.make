PROGRAM(openssl)



LICENSE(OpenSSL SSLeay)

PEERDIR(
    contrib/libs/openssl
    contrib/libs/openssl/crypto
)

ADDINCL(
    contrib/libs/openssl
    contrib/libs/openssl/apps
    contrib/libs/openssl/include
)

NO_COMPILER_WARNINGS()

NO_RUNTIME()

CFLAGS(
    -DAESNI_ASM
    -DECP_NISTZ256_ASM
    -DKECCAK1600_ASM
    -DOPENSSL_BN_ASM_MONT
    -DOPENSSL_CPUID_OBJ
    -DOPENSSL_PIC
    -DPOLY1305_ASM
    -DSHA1_ASM
    -DSHA256_ASM
    -DSHA512_ASM
    -DVPAES_ASM
    -DZLIB
)

IF (OS_DARWIN AND ARCH_X86_64 OR OS_LINUX AND ARCH_AARCH64 OR OS_LINUX AND ARCH_X86_64)
    CFLAGS(
        -DENGINESDIR=\"/usr/local/lib/engines-1.1\"
        -DOPENSSLDIR=\"/usr/local/ssl\"
    )
ENDIF()

IF (OS_DARWIN AND ARCH_X86_64 OR OS_LINUX AND ARCH_X86_64 OR OS_WINDOWS AND ARCH_X86_64)
    CFLAGS(
        -DGHASH_ASM
        -DL_ENDIAN
        -DMD5_ASM
        -DOPENSSL_BN_ASM_GF2m
        -DOPENSSL_BN_ASM_MONT5
        -DOPENSSL_IA32_SSE2
        -DPADLOCK_ASM
        -DRC4_ASM
        -DX25519_ASM
    )
ENDIF()

IF (OS_LINUX AND ARCH_AARCH64 OR OS_LINUX AND ARCH_X86_64)
    CFLAGS(-DOPENSSL_USE_NODELETE)
ENDIF()

IF (OS_DARWIN AND ARCH_X86_64)
    CFLAGS(
        -D_REENTRANT
    )
ENDIF()

IF (OS_DARWIN AND ARCH_ARM64)
    CFLAGS(
        -DL_ENDIAN
        -DOPENSSL_PIC
        -D_REENTRANT
    )
ENDIF()

IF (OS_WINDOWS)
    IF (ARCH_X86_64)
        CFLAGS(
            -DENGINESDIR="\"C:\\\\Program\ Files\\\\OpenSSL\\\\lib\\\\engines-1_1\""
            -DOPENSSLDIR="\"C:\\\\Program\ Files\\\\Common\ Files\\\\SSL\""
        )
    ELSEIF(ARCH_I386)
        CFLAGS(
            -DENGINESDIR="\"C:\\\\Program\ Files\ \(x86\)\\\\OpenSSL\\\\lib\\\\engines-1_1\""
            -DOPENSSLDIR="\"C:\\\\Program\ Files\ \(x86\)\\\\Common\ Files\\\\SSL\""
        )
    ENDIF()

    CFLAGS(
        -DOPENSSL_SYS_WIN32
        -DUNICODE
        -DWIN32_LEAN_AND_MEAN
        -D_CRT_SECURE_NO_DEPRECATE
        -D_UNICODE
        -D_WINSOCK_DEPRECATED_NO_WARNINGS
        /GF
    )
ENDIF()

SRCS(
    app_rand.c
    apps.c
    asn1pars.c
    bf_prefix.c
    ca.c
    ciphers.c
    cms.c
    crl.c
    crl2p7.c
    dgst.c
    dhparam.c
    dsa.c
    dsaparam.c
    ec.c
    ecparam.c
    enc.c
    engine.c
    errstr.c
    gendsa.c
    genpkey.c
    genrsa.c
    nseq.c
    ocsp.c
    openssl.c
    opt.c
    passwd.c
    pkcs12.c
    pkcs7.c
    pkcs8.c
    pkey.c
    pkeyparam.c
    pkeyutl.c
    prime.c
    rand.c
    rehash.c
    req.c
    rsa.c
    rsautl.c
    s_cb.c
    s_client.c
    s_server.c
    s_socket.c
    s_time.c
    sess_id.c
    smime.c
    speed.c
    spkac.c
    srp.c
    storeutl.c
    ts.c
    verify.c
    version.c
    x509.c
)

IF (OS_WINDOWS)
    SRCS(
        win32_init.c
    )
ENDIF()

END()
