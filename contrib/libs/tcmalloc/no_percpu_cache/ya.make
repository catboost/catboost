LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(Apache-2.0)



SRCDIR(contrib/libs/tcmalloc)

GLOBAL_SRCS(
    # Options
    tcmalloc/want_hpaa.cc
)

INCLUDE(../common.inc)

SRCS(aligned_alloc.c)

CFLAGS(
    -DTCMALLOC_256K_PAGES
    -DTCMALLOC_DEPRECATED_PERTHREAD
)

END()
