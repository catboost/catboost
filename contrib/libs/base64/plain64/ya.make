

LIBRARY()

LICENSE(
    BSD-2-Clause
    MIT
)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

NO_UTIL()

SRCS(
    codec_plain.c
    lib.c
)

IF (OS_LINUX OR OS_DARWIN)
    CONLYFLAGS(-std=c11)
ENDIF()

END()
