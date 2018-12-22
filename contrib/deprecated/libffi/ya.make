LIBRARY()



LICENSE(MIT)

NO_UTIL()
NO_COMPILER_WARNINGS()

ADDINCL(
    contrib/deprecated/libffi/include
)

IF (OS_WINDOWS)
    CFLAGS(
        GLOBAL -DFFI_BUILDING
    )
ENDIF()

SRCS(
    src/closures.c
    src/prep_cif.c
    src/raw_api.c
    src/types.c
)

IF (OS_IOS)
    SRCS(
        src/dlmalloc.c
    )
ELSE()
    SRCS(
        src/java_raw_api.c
    )
ENDIF()

IF (OS_IOS)
    IF(ARCH_ARM64)
        SRCS(
            src/aarch64/ffi_arm64.c
            src/aarch64/sysv_arm64.S
        )
    ELSEIF(ARCH_ARM7)
        SRCS(
            src/arm/ffi_armv7.c
            src/arm/sysv_armv7.S
            src/arm/trampoline_armv7.S
        )
    ELSEIF(ARCH_I386)
        SRCS(
            src/x86/ffi_i386.c
            src/x86/darwin_i386.S
        )
    ELSEIF(ARCH_X86_64)
        SRCS(
            src/x86/ffi64_x86_64.c
            src/x86/darwin64_x86_64.S
        )
    ENDIF()
ELSEIF (OS_WINDOWS)
    SRCS(
        src/x86/ffi.c
    )
    IF (ARCH_I386)
        SRCS(
            src/x86/win32.masm
        )
    ELSE()
        SRCS(
            src/x86/win64.masm
        )
    ENDIF()
ELSEIF (ARCH_X86_64)
    SRCS(
        src/x86/ffi.c
    )

    IF (OS_LINUX)
        SRCS(
            src/x86/ffi64.c
            src/x86/sysv.S
            src/x86/unix64.S
        )
    ENDIF()

    IF (OS_FREEBSD)
        SRCS(
            src/x86/freebsd.S
        )
    ENDIF()

    IF (OS_DARWIN)
        SRCS(
            src/x86/darwin.S
            src/x86/darwin64.S
            src/x86/ffi64.c
        )
    ENDIF()
ENDIF()

END()
