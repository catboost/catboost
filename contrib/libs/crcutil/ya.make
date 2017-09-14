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
    IF (OS_LINUX OR OS_DARWIN OR OS_IOS OR OS_FREEBSD OR CYGWIN)
        IF (CLANG)
            CFLAGS(
                -DCRCUTIL_USE_MM_CRC32=1
            )
        ELSE()
            CFLAGS(
                -mcrc32 -DCRCUTIL_USE_MM_CRC32=1
            )
        ENDIF()

        SRCS(
            multiword_128_64_gcc_amd64_sse2.cc
            multiword_64_64_gcc_amd64_asm.cc
            multiword_64_64_gcc_i386_mmx.cc
        )
    ENDIF()

    IF (OS_WINDOWS)
        SRCS(
            multiword_64_64_cl_i386_mmx.cc
        )
    ENDIF()
ENDIF ()

SRCS(
    crc32c_sse4.cc
    interface.cc
    multiword_64_64_intrinsic_i386_mmx.cc
)

END()
