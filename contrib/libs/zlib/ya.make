LIBRARY()


NO_UTIL()
NO_COMPILER_WARNINGS()
NO_JOIN_SRC()

CFLAGS(
    -DBUILD_ZLIB
)

IF (BUILD_AS_ORIGIN)
    CFLAGS(-DY_BUILD_AS_ORIGIN)
ENDIF ()

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
