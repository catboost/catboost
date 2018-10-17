LIBRARY()



PEERDIR(
    library/python/runtime
    library/python/symbols/module
    library/python/symbols/libc
    library/python/symbols/uuid
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
