LIBRARY()


NO_UTIL()
NO_COMPILER_WARNINGS()
JOINSRC()

IF (SANITIZER_TYPE STREQUAL undefined)
    NO_SANITIZE()
ENDIF ()

SRCS(
    randtable.c
    crctable.c
    compress.c
    bzlib.c
    decompress.c
    blocksort.c
    huffman.c
)

END()
