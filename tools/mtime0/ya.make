DLL_TOOL(mtime0 PREFIX "")



EXPORTS_SCRIPT(mtime0.exports)

NO_RUNTIME()

CFLAGS(
    -fPIC
)

IF (OS_LINUX)
    SRCS(
        mtime0.c
    )
ENDIF()

END()
