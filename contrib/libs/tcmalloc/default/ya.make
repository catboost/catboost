LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(Apache-2.0)



SRCDIR(contrib/libs/tcmalloc)

INCLUDE(../common.inc)

GLOBAL_SRCS(
    # Options
    tcmalloc/want_hpaa_subrelease.cc
)

END()
