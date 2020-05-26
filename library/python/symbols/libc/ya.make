LIBRARY()



PEERDIR(
    library/python/symbols/registry
)

IF (GCC OR CLANG)
    CFLAGS(
        -Wno-deprecated-declarations # For sem_getvalue.
    )
ENDIF()

SRCS(
    GLOBAL syms.cpp
)

END()
