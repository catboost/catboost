

LIBRARY()

LICENSE(BSD-3-Clause)

NO_COMPILER_WARNINGS()
NO_UTIL()
NO_RUNTIME()

IF (NOT MUSL AND NOT OS_FREEBSD AND NOT OS_DARWIN AND NOT OS_IOS)
    SRCS(
        strlcat.c
        strlcpy.c
    )
    IF (NOT OS_LINUX)
        SRCS(
            stpcpy.c
        )
    ENDIF()
ENDIF()

IF (NOT OS_WINDOWS)
    SRCS(
        string.c
    )
ENDIF()

# NB: nested IF's are needed due to the lack of lazy evaluation of logical statements: DEVTOOLS-7837
IF (OS_ANDROID)
    IF (ANDROID_API < 28)
        SRCS(
            glob.c
            reallocarray.c
        )
    ENDIF()
ENDIF()

IF (OS_WINDOWS OR OS_DARWIN OR OS_IOS)
    SRCS(
       memrchr.c
    )
ENDIF()

IF (OS_WINDOWS)
    ADDINCL(GLOBAL contrib/libs/libc_compat/include/windows)

    SRCS(
        strcasestr.c
        strsep.c
        src/windows/sys/uio.c
    )
ENDIF()

IF (NOT MUSL AND OS_LINUX AND OS_SDK STREQUAL "ubuntu-12")
    ADDINCL(GLOBAL contrib/libs/libc_compat/include/uchar)
ENDIF()

END()
