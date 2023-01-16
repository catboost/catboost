LIBRARY()

LICENSE(MIT)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

VERSION(2016)

ORIGINAL_SOURCE(https://www.nayuki.io/page/fast-md5-hash-implementation-in-x86-assembly)



IF (OS_LINUX AND ARCH_X86_64)
    SRCS(
        md5-fast-x8664.S
    )
ELSE()
    SRCS(
        md5.c
    )
ENDIF()

END()
