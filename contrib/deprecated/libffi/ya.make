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
    src/prep_cif.c
    src/types.c
    src/raw_api.c
    src/java_raw_api.c
    src/closures.c
)

IF (ARCH_X86_64)
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

    IF (OS_WINDOWS)
        SRCS(
            src/x86/win64.masm
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
