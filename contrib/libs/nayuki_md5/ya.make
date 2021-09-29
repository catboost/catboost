LIBRARY()

LICENSE(MIT)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)



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
