

LIBRARY()

LICENSE(
    BSD-1-Clause
    BSD-2-Clause
    BSD-3-Clause
    ISC
)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

NO_COMPILER_WARNINGS()

NO_UTIL()

NO_RUNTIME()

IF (NOT OS_WINDOWS)
    SRCS(
        string.c
    )
ENDIF()

# Android libc function appearance is documented here:
# https://android.googlesource.com/platform/bionic/+/master/docs/status.md
#
# NB: nested IF's are needed due to the lack of lazy evaluation of logical statements: DEVTOOLS-7837
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
    IF (ANDROID_API < 24)
        SRCS(
            ifaddrs.c
        )
        ADDINCL(
            GLOBAL contrib/libs/libc_compat/include/ifaddrs
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

IF (OS_DARWIN)
    SRCS(
        reallocarray.c
    )
ENDIF()

IF (OS_WINDOWS)
    ADDINCL(
        GLOBAL contrib/libs/libc_compat/include/windows
    )
    SRCS(
        reallocarray.c
        stpcpy.c
        strlcat.c
        strlcpy.c
        strcasestr.c
        strsep.c
        src/windows/sys/uio.c
    )
ENDIF()

IF (OS_LINUX)
    ADDINCL(
        GLOBAL contrib/libs/libc_compat/include/readpassphrase
    )
    SRCS(
        readpassphrase.c
    )
ENDIF()

IF (OS_LINUX AND NOT MUSL)
    IF (OS_SDK == "ubuntu-12")
        ADDINCL(
            # uchar.h was introduced in glibc=2.16
            GLOBAL contrib/libs/libc_compat/include/uchar
        )
    ENDIF()
    IF (OS_SDK == "ubuntu-12" OR OS_SDK == "ubuntu-14" OR OS_SDK == "ubuntu-16")
        ADDINCL(
            GLOBAL contrib/libs/libc_compat/include/random
        )
        SRCS(
            # getrandom was added in glibc=2.25
            getrandom.c
            # memfd_create was added in glibc=2.27
            memfd_create.c
        )
    ENDIF()
    IF (OS_SDK != "ubuntu-20")
        SRCS(
            # reallocarray was added in glibc=2.29
            reallocarray.c
        )
    ENDIF()
    SRCS(
        # glibc does not offer strlcat / strlcpy yet
        strlcat.c
        strlcpy.c
    )
ENDIF()

END()
