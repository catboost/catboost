LIBRARY()

LICENSE(
    PD
)



CFLAGS(-D_7ZIP_ST=1)

NO_UTIL()

NO_WSHADOW()

SRCS(
    7zStream.c
    Alloc.c
    LzmaDec.c
    LzmaEnc.c
    LzFind.c
    LzmaLib.c
    LzmaUtil.c
)

END()
