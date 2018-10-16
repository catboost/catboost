LIBRARY()



PEERDIR(
    library/python/runtime
    library/python/symbols/module
    library/python/libc
)

IF (MUSL)
    PEERDIR(
        library/python/pythonapi
    )
ENDIF()

USE_PYTHON()

SRCS(
    main.c
)

END()

NEED_CHECK()
