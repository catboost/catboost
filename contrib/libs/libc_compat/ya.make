

LIBRARY()

LICENSE(
    BSD-1-Clause AND
    BSD-2-Clause AND
    BSD-3-Clause AND
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

DISABLE(PROVIDE_GETRANDOM_GETENTROPY)
DISABLE(PROVIDE_REALLOCARRAY)

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
        )
        ENABLE(PROVIDE_REALLOCARRAY)
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
        explicit_bzero.c
    )
    ENABLE(PROVIDE_REALLOCARRAY)
ENDIF()

IF (OS_WINDOWS)
    ADDINCL(
        GLOBAL contrib/libs/libc_compat/include/windows
    )
    SRCS(
        explicit_bzero.c
        stpcpy.c
        strlcat.c
        strlcpy.c
        strcasestr.c
        strsep.c
        src/windows/sys/uio.c
    )
    ENABLE(PROVIDE_REALLOCARRAY)
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
        # getrandom and getentropy were added in glibc=2.25
        ENABLE(PROVIDE_GETRANDOM_GETENTROPY)

        SRCS(
            # explicit_bzero was added in glibc=2.25
            explicit_bzero.c
            # memfd_create was added in glibc=2.27
            memfd_create.c
        )
    ENDIF()
    IF (OS_SDK != "ubuntu-20")
        # reallocarray was added in glibc=2.29
        ENABLE(PROVIDE_REALLOCARRAY)
    ENDIF()
    SRCS(
        # glibc does not offer strlcat / strlcpy yet
        strlcat.c
        strlcpy.c
    )
    IF (SANITIZER_TYPE == "memory")
        # llvm sanitized runtime is missing an interceptor for a buggy (getservbyname{_r}).
        # See: https://github.com/google/sanitizers/issues/1138
        ENABLE(PROVIDE_GETSERVBYNAME)
    ENDIF()
ENDIF()

IF (PROVIDE_REALLOCARRAY)
    SRCS(
        reallocarray/reallocarray.c
    )
    ADDINCL(
        ONE_LEVEL contrib/libs/libc_compat/reallocarray
    )
ENDIF()

IF (PROVIDE_GETRANDOM_GETENTROPY)
    SRCS(
        random/getrandom.c
        random/getentropy.c
    )
    ADDINCL(
        ONE_LEVEL contrib/libs/libc_compat/random
    )
ENDIF()

IF (PROVIDE_GETSERVBYNAME)
    SRCS(
        getservbyname/getservbyname.c
        getservbyname/getservbyname_r.c
        getservbyname/lookup_serv.c
    )
ENDIF()

END()
