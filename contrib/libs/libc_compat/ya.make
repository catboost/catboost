

LIBRARY()

LICENSE(BSD)

NO_COMPILER_WARNINGS()
NO_UTIL()
NO_RUNTIME()

IF (NOT MUSL AND NOT OS_FREEBSD)
    SRCS(
        strlcat.c
        strlcpy.c
    )
ENDIF()

END()
