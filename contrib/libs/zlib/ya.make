LIBRARY()

LICENSE(ZLIB)



NO_UTIL()
NO_COMPILER_WARNINGS()
NO_JOIN_SRC()

SRCS(
    adler32.c
    compress.c
    crc32.c
    deflate.c
    gzclose.c
    gzlib.c
    gzread.c
    gzwrite.c
    infback.c
    inffast.c
    inflate.c
    inftrees.c
    trees.c
    uncompr.c
    zutil.c
)

END()
