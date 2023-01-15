LIBRARY()

LICENSE(
    APACHE2
)



NO_UTIL()
NO_COMPILER_WARNINGS()
NO_JOIN_SRC()

IF (GCC AND USE_LTO)
    CFLAGS(-DCRCUTIL_FORCE_ASM_CRC32C=1)
ENDIF ()

IF (ARCH_I386 OR ARCH_X86_64)
    IF (OS_WINDOWS)
        SRCS(
            multiword_64_64_cl_i386_mmx.cc
        )
    ELSEIF(OS_ANDROID AND ARCH_I386)
        # 32-bit Android has some problems with register allocation, so we fall back to default implementation
    ELSE()
        IF (CLANG)
            CFLAGS(
                -DCRCUTIL_USE_MM_CRC32=1
            )
            IF (ARCH_I386)
                # clang doesn't support this as optimization attribute and has problems with register allocation
                SRC(multiword_64_64_gcc_i386_mmx.cc -fomit-frame-pointer)
            ELSE()
                SRCS(multiword_64_64_gcc_i386_mmx.cc)
            ENDIF()
        ELSE()
            CFLAGS(
                -mcrc32 -DCRCUTIL_USE_MM_CRC32=1
            )
        ENDIF()

        SRCS(
            multiword_128_64_gcc_amd64_sse2.cc
            multiword_64_64_gcc_amd64_asm.cc
        )
    ENDIF()

    IF (OS_WINDOWS)
        SRCS(crc32c_sse4.cc)
    ELSE()
        SRC_CPP_SSE4(crc32c_sse4.cc)
    ENDIF()
ENDIF ()

SRCS(
    interface.cc
    multiword_64_64_intrinsic_i386_mmx.cc
)

END()
