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

IF (ARCH_X86_64)
    IF (OS_LINUX)
        SRCS(
            src/x86/ffi.c
            src/x86/ffi64.c
            src/x86/sysv.S
            src/x86/unix64.S
        )
    ELSEIF (OS_WINDOWS)
        SRCS(
            src/x86/ffi.c
            src/x86/win64.masm
        )
    ELSEIF (OS_DARWIN)
        SRCS(
            src/x86/ffi.c
            src/x86/darwin.S
            src/x86/darwin64.S
            src/x86/ffi64.c
        )
    ELSEIF (OS_IOS)
        SRCS(
            src/x86/ffi64_x86_64.c
            src/x86/darwin64_x86_64.S
        )
    ELSEIF (OS_ANDROID)
        MESSAGE(WARNING Unsupported libffi platform: Android on x86_64. Linking of final executable will fail)
    ELSE()
        MESSAGE(FATAL_ERROR Unsupported libffi OS for x86_64: ${TARGET_PLATFORM})
    ENDIF()
ELSEIF (ARCH_I386)
    IF (OS_WINDOWS)
        SRCS(
            src/x86/ffi.c
            src/x86/win32.masm
        )
    ELSEIF (OS_IOS)
        SRCS(
            src/x86/ffi_i386.c
            src/x86/darwin_i386.S
        )
    ELSEIF (OS_ANDROID)
        MESSAGE(WARNING Unsupported libffi platform: Android on i386. Linking of final executable will fail)
    ELSE()
        MESSAGE(FATAL_ERROR Unsupported libffi OS for i386: ${TARGET_PLATFORM})
    ENDIF()
ELSEIF (ARCH_ARM64)
    IF (OS_IOS)
        SRCS(
            src/aarch64/ffi_arm64.c
            src/aarch64/sysv_arm64.S
        )
    ELSEIF (OS_ANDROID)
        MESSAGE(WARNING Unsupported libffi platform: Android on armv8/aarch64/arm64. Linking of final executable will fail)
    ELSE()
        MESSAGE(FATAL_ERROR Unsupported libffi OS for armv8/aarch64/arm64: ${TARGET_PLATFORM})
    ENDIF()
ELSEIF (ARCH_ARM7)
    IF (OS_IOS)
        SRCS(
            src/arm/ffi_armv7.c
            src/arm/sysv_armv7.S
            src/arm/trampoline_armv7.S
        )
    ELSEIF (OS_ANDROID)
        MESSAGE(WARNING Unsupported libffi platform: Android on armv7. Linking of final executable will fail)
    ELSE()
        MESSAGE(FATAL_ERROR Unsupported libffi OS for armv7: ${TARGET_PLATFORM})
    ENDIF()
ELSE()
    MESSAGE(FATAL_ERROR Unsupported libffi ARCH: ${HARDWARE_TYPE})
ENDIF()

END()
