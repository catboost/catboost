LIBRARY()



VERSION(18.00) # https://www.7-zip.org/sdk.html

LICENSE(PD)

CFLAGS(-D_7ZIP_ST=1)

NO_UTIL()

SRCS(
    7zStream.c
    Alloc.c
    LzmaDec.c
    LzmaEnc.c
    LzFind.c
    LzmaLib.c
)

END()
