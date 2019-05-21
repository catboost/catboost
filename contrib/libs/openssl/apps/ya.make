PROGRAM(openssl)

LICENSE(
    OpenSSL
    SSLeay
)



NO_COMPILER_WARNINGS()

NO_UTIL()

PEERDIR(
    contrib/libs/openssl
)

ADDINCL(
    contrib/libs/openssl
    contrib/libs/openssl/apps
    contrib/libs/openssl/include
)

CFLAGS(
    -DECP_NISTZ256_ASM
    -DKECCAK1600_ASM
    -DOPENSSL_BN_ASM_MONT
    -DOPENSSL_CPUID_OBJ
    -DPOLY1305_ASM
    -DSHA1_ASM
    -DSHA256_ASM
    -DSHA512_ASM
    -DVPAES_ASM
)

IF (OS_DARWIN AND ARCH_X86_64 OR OS_LINUX AND ARCH_AARCH64 OR OS_LINUX AND ARCH_X86_64)
    CFLAGS(
        -DENGINESDIR=\"/usr/local/lib/engines-1.1\"
        -DOPENSSLDIR=\"/usr/local/ssl\"
    )
ENDIF()

IF (OS_DARWIN AND ARCH_X86_64 OR OS_LINUX AND ARCH_X86_64 OR OS_WINDOWS AND ARCH_X86_64)
    CFLAGS(
        -DAES_ASM
        -DBSAES_ASM
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

IF (OS_WINDOWS AND ARCH_X86_64)
    CFLAGS(
        -DENGINESDIR="\"C:\\\\Program\ Files\\\\OpenSSL\\\\lib\\\\engines-1_1\""
        -DOPENSSLDIR="\"C:\\\\Program\ Files\\\\Common\ Files\\\\SSL\""
        -DOPENSSL_SYS_WIN32
        -DUNICODE
        -DWIN32_LEAN_AND_MEAN
        -D_CRT_SECURE_NO_DEPRECATE
        -D_UNICODE
        -D_WINSOCK_DEPRECATED_NO_WARNINGS
        /GF
    )
ENDIF()

IF (OS_WINDOWS AND ARCH_X86_64)
    LDFLAGS(
        advapi32.lib
        crypt32.lib
        gdi32.lib
        setargv.obj
        user32.lib
        ws2_32.lib
    )
ENDIF()

SRCDIR(contrib/libs/openssl)

SRCS(
    apps/app_rand.c
    apps/apps.c
    apps/asn1pars.c
    apps/bf_prefix.c
    apps/ca.c
    apps/ciphers.c
    apps/cms.c
    apps/crl.c
    apps/crl2p7.c
    apps/dgst.c
    apps/dhparam.c
    apps/dsa.c
    apps/dsaparam.c
    apps/ec.c
    apps/ecparam.c
    apps/enc.c
    apps/engine.c
    apps/errstr.c
    apps/gendsa.c
    apps/genpkey.c
    apps/genrsa.c
    apps/nseq.c
    apps/ocsp.c
    apps/openssl.c
    apps/opt.c
    apps/passwd.c
    apps/pkcs12.c
    apps/pkcs7.c
    apps/pkcs8.c
    apps/pkey.c
    apps/pkeyparam.c
    apps/pkeyutl.c
    apps/prime.c
    apps/rand.c
    apps/rehash.c
    apps/req.c
    apps/rsa.c
    apps/rsautl.c
    apps/s_cb.c
    apps/s_client.c
    apps/s_server.c
    apps/s_socket.c
    apps/s_time.c
    apps/sess_id.c
    apps/smime.c
    apps/speed.c
    apps/spkac.c
    apps/srp.c
    apps/storeutl.c
    apps/ts.c
    apps/verify.c
    apps/version.c
    apps/x509.c
)

IF (OS_WINDOWS AND ARCH_X86_64)
    SRCS(
        apps/win32_init.c
    )
ENDIF()

END()
