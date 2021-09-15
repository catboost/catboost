LIBRARY()

LICENSE(Public-Domain)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)



# https://www.7-zip.org/sdk.html
VERSION(19.00)

CFLAGS(-D_7ZIP_ST=1)

NO_UTIL()

SRCS(
    7zStream.c
    Aes.c
    AesOpt.c
    Alloc.c
    Bra.c
    Bra86.c
    BraIA64.c
    CpuArch.c
    LzFind.c
    Lzma2Dec.c
    Lzma2Enc.c
    LzmaDec.c
    LzmaEnc.c
    LzmaLib.c
    Sha256.c
)

END()
