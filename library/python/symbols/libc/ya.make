LIBRARY()



PEERDIR(
    library/python/symbols/registry
)

IF (OS_DARWIN)
    CFLAGS(
        -Wno-deprecated-declarations # For sem_getvalue.
    )
ENDIF()

SRCS(
    GLOBAL syms.cpp
)

END()
