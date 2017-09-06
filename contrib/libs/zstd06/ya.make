LIBRARY()

LICENSE(
    BSD
)



NO_UTIL()

SRCS(
    common/entropy_common.c
    common/fse_decompress.c
    common/xxhash.c
    common/zstd_common.c
    compress/fse_compress.c
    compress/huf_compress.c
    compress/zbuff_compress.c
    compress/zstd_compress.c
    decompress/huf_decompress.c
    decompress/zbuff_decompress.c
    decompress/zstd_decompress.c
    dictBuilder/divsufsort.c
    dictBuilder/zdict.c
    legacy/zstd_v01.c
    legacy/zstd_v02.c
    legacy/zstd_v03.c
    legacy/zstd_v04.c
    legacy/zstd_v05.c
    legacy/zstd_v07.c
    legacy/zstd_v08.c
)

NO_COMPILER_WARNINGS()

CFLAGS(-DZSTD_LEGACY_SUPPORT=1)

ADDINCL(
    contrib/libs/zstd06
    contrib/libs/zstd06/common
    contrib/libs/zstd06/compress
    contrib/libs/zstd06/decompress
    contrib/libs/zstd06/dictBuilder
    contrib/libs/zstd06/legacy
)

END()
