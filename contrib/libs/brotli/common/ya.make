LIBRARY()

LICENSE(
    MIT
)



NO_UTIL()
NO_COMPILER_WARNINGS()

ADDINCL(
    contrib/libs/brotli/include
)

SRCS(
    dictionary.c
    transform.c
)

CFLAGS(
    -DBROTLI_BUILD_PORTABLE
)

END()
