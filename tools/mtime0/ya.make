DLL_TOOL(mtime0 PREFIX "")



NO_RUNTIME()

IF (OS_LINUX)
    EXPORTS_SCRIPT(mtime0.exports)

    CFLAGS(
        -fPIC
    )

    SRCS(
        mtime0.c
    )
ENDIF()

END()
