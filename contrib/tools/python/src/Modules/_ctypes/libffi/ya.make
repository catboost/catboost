

LIBRARY()

NO_UTIL()

NO_COMPILER_WARNINGS()

IF (OS_WINDOWS)
    ADDINCL(GLOBAL contrib/tools/python/src/Modules/_ctypes/libffi_msvc)
    SRCDIR(contrib/tools/python/src/Modules/_ctypes/libffi_msvc)
    CFLAGS(GLOBAL /DX86_WIN64)
    SRCS(
        prep_cif.c
        ffi.c
        win64.masm
    )
ELSE()
    ADDINCL(GLOBAL contrib/tools/python/src/Modules/_ctypes/libffi/include)
    ADDINCL(GLOBAL contrib/tools/python/src/Modules/_ctypes/libffi/src/x86)
    SRCDIR(contrib/tools/python/src/Modules/_ctypes/libffi/src)
    CFLAGS(-DHAVE_CONFIG_H)
    SRCS(
        debug.c
        prep_cif.c
        types.c
        raw_api.c
        java_raw_api.c
        closures.c
        src/x86/ffi.c
        src/x86/ffi64.c
    )
    IF (OS_DARWIN)
        CFLAGS(GLOBAL -DX86_DARWIN)
        SRCS(
            x86/darwin64.S
            x86/darwin.S
        )
    ELSE()
        CFLAGS(GLOBAL -DX86_64)
        SRCS(
            src/x86/sysv.S
            src/x86/unix64.S
        )
    ENDIF()
ENDIF()

END()

