

# This is a part of xz utils package. Source can be downloaded from
# https://tukaani.org/xz/

LIBRARY()

LICENSE(Public-Domain)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

VERSION(5.2.4)

CFLAGS(-DTUKLIB_SYMBOL_PREFIX=lzma_)

ADDINCL(
    contrib/libs/xz
    contrib/libs/xz/common
)

SRCS(
    tuklib_cpucores.c
)

END()
