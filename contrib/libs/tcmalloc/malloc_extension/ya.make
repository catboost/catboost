LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(Apache-2.0)



NO_UTIL()

NO_COMPILER_WARNINGS()

# https://github.com/google/tcmalloc
VERSION(2020-11-23-a643d89610317be1eff9f7298104eef4c987d8d5)

SRCDIR(contrib/libs/tcmalloc)

SRCS(
    tcmalloc/malloc_extension.cc
)

PEERDIR(
    contrib/restricted/abseil-cpp
)

ADDINCL(GLOBAL contrib/libs/tcmalloc)

CFLAGS(-DTCMALLOC_256K_PAGES)

END()
