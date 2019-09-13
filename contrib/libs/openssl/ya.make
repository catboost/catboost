LIBRARY()

LICENSE(
    OpenSSL
    SSLeay
)



NO_COMPILER_WARNINGS()

NO_UTIL()

PEERDIR(
    contrib/libs/openssl/crypto
    contrib/libs/zlib
)

ADDINCL(
    contrib/libs/openssl
    contrib/libs/openssl/crypto
    contrib/libs/openssl/crypto/ec/curve448
    contrib/libs/openssl/crypto/ec/curve448/arch_32
    contrib/libs/openssl/crypto/include
    contrib/libs/openssl/crypto/modes
    contrib/libs/openssl/include
    contrib/libs/zlib
    GLOBAL contrib/libs/openssl/include
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

IF (NOT OS_WINDOWS)
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

IF (SANITIZER_TYPE STREQUAL memory)
    CFLAGS(-DPURIFY)
ENDIF()

IF (MUSL)
    CFLAGS(-DOPENSSL_NO_ASYNC)
ENDIF()

IF (ARCH_TYPE_32)
    CFLAGS(-DOPENSSL_NO_EC_NISTP_64_GCC_128)
ENDIF()

SRCS(
    engines/e_capi.c
    engines/e_padlock.c
    ssl/bio_ssl.c
    ssl/d1_lib.c
    ssl/d1_msg.c
    ssl/d1_srtp.c
    ssl/methods.c
    ssl/packet.c
    ssl/pqueue.c
    ssl/record/dtls1_bitmap.c
    ssl/record/rec_layer_d1.c
    ssl/record/rec_layer_s3.c
    ssl/record/ssl3_buffer.c
    ssl/record/ssl3_record.c
    ssl/record/ssl3_record_tls13.c
    ssl/s3_cbc.c
    ssl/s3_enc.c
    ssl/s3_lib.c
    ssl/s3_msg.c
    ssl/ssl_asn1.c
    ssl/ssl_cert.c
    ssl/ssl_ciph.c
    ssl/ssl_conf.c
    ssl/ssl_err.c
    ssl/ssl_init.c
    ssl/ssl_lib.c
    ssl/ssl_mcnf.c
    ssl/ssl_rsa.c
    ssl/ssl_sess.c
    ssl/ssl_stat.c
    ssl/ssl_txt.c
    ssl/ssl_utst.c
    ssl/statem/extensions.c
    ssl/statem/extensions_clnt.c
    ssl/statem/extensions_cust.c
    ssl/statem/extensions_srvr.c
    ssl/statem/statem.c
    ssl/statem/statem_clnt.c
    ssl/statem/statem_dtls.c
    ssl/statem/statem_lib.c
    ssl/statem/statem_srvr.c
    ssl/t1_enc.c
    ssl/t1_lib.c
    ssl/t1_trce.c
    ssl/tls13_enc.c
    ssl/tls_srp.c
)

IF (OS_LINUX AND ARCH_AARCH64 OR OS_LINUX AND ARCH_X86_64 OR OS_LINUX AND ARCH_PPC64LE)
    SRCS(
        engines/e_afalg.c
    )
ENDIF()

IF (OS_DARWIN AND ARCH_X86_64)
    SRCS(
        asm/darwin/engines/e_padlock-x86_64.s
    )
ENDIF()

IF (OS_LINUX AND ARCH_X86_64)
    SRCS(
        asm/linux/engines/e_padlock-x86_64.s
    )
ENDIF()

IF (OS_WINDOWS AND ARCH_X86_64)
    SRCS(
        asm/windows/engines/e_padlock-x86_64.asm
    )
ENDIF()

END()
