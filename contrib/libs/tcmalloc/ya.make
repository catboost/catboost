LIBRARY()

LICENSE(
    Apache-2.0
    LicenseRef-scancode-other-permissive
)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)



# https://github.com/google/tcmalloc
VERSION(2021-07-30-ae1b63023d69c188b6de4e9f14fb09e30f241d55)

SRCS(
    # Options
    tcmalloc/want_hpaa.cc
)

INCLUDE(common.inc)

CFLAGS(-DTCMALLOC_256K_PAGES)

END()

