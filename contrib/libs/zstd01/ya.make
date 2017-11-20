LIBRARY()

LICENSE(
    BSD
)

IF (SANITIZER_TYPE STREQUAL "undefined")
    NO_SANITIZE()
ENDIF ()



SRCS(
    fse.c
    zstd.c
)

END()
