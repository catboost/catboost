

LIBRARY()

LICENSE(BSD-3-Clause)

NO_COMPILER_WARNINGS()
NO_UTIL()
NO_RUNTIME()

IF (NOT OS_WINDOWS)
    SRCS(
        string.c
    )
ENDIF()

IF (OS_LINUX)
    IF (NOT MUSL)
        SRCS(
            strlcat.c
            strlcpy.c
        )
    ENDIF()
ENDIF()

IF (OS_ANDROID)
    SRCS(
        strlcat.c
        strlcpy.c
    )
    IF (ANDROID_API < 28)
        SRCS(
            glob.c
            reallocarray.c
        )
    ENDIF()
    IF (ANDROID_API < 21)
        SRCS(
            stpcpy.c
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
        stpcpy.c
        strlcat.c
        strlcpy.c
        strcasestr.c
        strsep.c
        src/windows/sys/uio.c
    )
ENDIF()

IF (NOT MUSL AND OS_LINUX AND OS_SDK STREQUAL "ubuntu-12")
    ADDINCL(GLOBAL contrib/libs/libc_compat/include/uchar)
ENDIF()

END()
