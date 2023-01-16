LIBRARY()



CFLAGS(
    -w
    -DMI_MALLOC_OVERRIDE=1
    -DMI_PADDING=0
)

LICENSE(MIT)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

VERSION(1.7.2)

ADDINCL(contrib/libs/mimalloc/include)

SRCS(
    src/static.c
)

END()
