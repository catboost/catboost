LIBRARY()

LICENSE(
    MIT
)



NO_UTIL()
NO_COMPILER_WARNINGS()

ADDINCL(
    GLOBAL contrib/libs/brotli/include
)

PEERDIR(
    contrib/libs/brotli/common
)

SRCS(
    bit_reader.c
    decode.c
    huffman.c
    state.c
)

END()
