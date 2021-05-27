LIBRARY()

LICENSE(OpenSSL SSLeay)



VERSION(1.1.1g)

PEERDIR(
    contrib/libs/openssl/crypto
)

ADDINCL(
    GLOBAL contrib/libs/openssl/include
    contrib/libs/openssl
)

IF (OS_LINUX)
    IF (ARCH_ARM64)
        SET(LINUX_ARM64 yes)
    ELSEIF(ARCH_ARM7)
        SET(LINUX_ARMV7 yes)
    ELSEIF(ARCH_X86_64)
        SET(LINUX_X86_64 yes)
    ENDIF()
ENDIF()

IF (OS_IOS)
    IF (ARCH_ARM64)
        SET(IOS_ARM64 yes)
    ELSEIF(ARCH_ARM7)
        SET(IOS_ARMV7 yes)
    ELSEIF(ARCH_X86_64)
        SET(IOS_X86_64 yes)
    ELSEIF(ARCH_I386)
        SET(IOS_I386 yes)
    ENDIF()
ENDIF()

IF (OS_ANDROID)
    IF (ARCH_ARM64)
        SET(ANDROID_ARM64 yes)
    ELSEIF(ARCH_ARM7)
        SET(ANDROID_ARMV7 yes)
    ELSEIF(ARCH_X86_64)
        SET(ANDROID_X86_64 yes)
    ELSEIF(ARCH_I686)
        SET(ANDROID_I686 yes)
    ENDIF()
ENDIF()

IF (OS_WINDOWS)
    IF (ARCH_X86_64)
        SET(WINDOWS_X86_64 yes)
    ELSEIF(ARCH_I686)
        SET(WINDOWS_I686 yes)
    ENDIF()
ENDIF()

NO_COMPILER_WARNINGS()

NO_RUNTIME()

CFLAGS(
    -DAESNI_ASM
    -DOPENSSL_BN_ASM_MONT
    -DOPENSSL_CPUID_OBJ
    -DSHA1_ASM
    -DSHA256_ASM
    -DSHA512_ASM
)

IF (NOT WINDOWS_I686)
    CFLAGS(
        -DECP_NISTZ256_ASM
        -DPOLY1305_ASM
    )
ENDIF()

IF (NOT ANDROID_I686 AND NOT WINDOWS_I686)
    CFLAGS(
        -DKECCAK1600_ASM
    )
ENDIF()

IF (NOT OS_WINDOWS)
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

IF (OS_LINUX AND ARCH_AARCH64 OR OS_LINUX AND ARCH_X86_64 OR OS_ANDROID)
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

IF (SANITIZER_TYPE == memory)
    CFLAGS(-DPURIFY)
ENDIF()

IF (MUSL)
    CFLAGS(-DOPENSSL_NO_ASYNC)
ENDIF()

IF (ARCH_TYPE_32)
    CFLAGS(-DOPENSSL_NO_EC_NISTP_64_GCC_128)
ENDIF()

SRCS(
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

IF (NOT IOS_ARMV7 AND NOT LINUX_ARMV7)
    CFLAGS(
        -DVPAES_ASM
    )
ENDIF()

IF (NOT IOS_ARM64 AND NOT IOS_ARMV7)
    SRCS(
        engines/e_capi.c
        engines/e_padlock.c
    )
ENDIF()

IF (OS_LINUX AND ARCH_ARM7 OR OS_LINUX AND ARCH_AARCH64 OR OS_LINUX AND ARCH_X86_64 OR OS_LINUX AND ARCH_PPC64LE)
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
        asm/windows/engines/e_padlock-x86_64.masm
    )
ENDIF()


IF (OS_WINDOWS AND ARCH_I386)
    CFLAGS(
        -DPADLOCK_ASM
    )

    SRCS(
        asm/windows/engines/e_padlock-x86.masm
    )
ENDIF()

IF (OS_IOS AND ARCH_X86_64)
    CFLAGS(
        -DPADLOCK_ASM
        -D_REENTRANT
    )
    SRCS(
        asm/ios/x86_64/engines/e_padlock-x86_64.s
        engines/e_dasync.c
        engines/e_ossltest.c
    )
ENDIF()

IF (OS_IOS AND ARCH_I386)
    CFLAGS(
        -DPADLOCK_ASM
        -D_REENTRANT
    )
    SRCS(
        asm/ios/i386/engines/e_padlock-x86.s
        engines/e_dasync.c
        engines/e_ossltest.c
    )
ENDIF()

IF (OS_ANDROID AND ARCH_X86_64)
    CFLAGS(
        -DOPENSSL_PIC
        -DOPENSSL_IA32_SSE2
        -DOPENSSL_BN_ASM_MONT5
        -DOPENSSL_BN_ASM_GF2m
        -DDRC4_ASM
        -DMD5_ASM
        -DGHASH_ASM
        -DX25519_ASM
    )
    SRCS(
        asm/android/x86_64/engines/e_padlock-x86_64.s
    )
ENDIF()

IF (OS_ANDROID AND ARCH_I686)
    CFLAGS(
        -DOPENSSL_PIC
        -DOPENSSL_BN_ASM_PART_WORDS
        -DOPENSSL_IA32_SSE2
        -DOPENSSL_BN_ASM_MONT
        -DOPENSSL_BN_ASM_GF2m
        -DRC4_ASM
        -DMD5_ASM
        -DRMD160_ASM
        -DVPAES_ASM
        -DWHIRLPOOL_ASM
        -DGHASH_ASM
    )
    SRCS(
        asm/android/i686/engines/e_padlock-x86.s
    )
ENDIF()

IF (OS_ANDROID AND ARCH_ARM7)
    CFLAGS(
        -DOPENSSL_PIC
        -DOPENSSL_BN_ASM_GF2m
        -DKECCAK1600_ASM
        -DAES_ASM
        -DBSAES_ASM
        -DGHASH_ASM
    )
ENDIF()

IF (OS_ANDROID AND ARCH_ARM64)
    CFLAGS(
        -DOPENSSL_PIC
        -DKECCAK1600_ASM
        -DVPAES_ASM
    )
ENDIF()

END()

IF (NOT DLL_FOR AND NOT OS_IOS)
    RECURSE(
    apps
    dynamic
)
ENDIF()
