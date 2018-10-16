LIBRARY()



PEERDIR(
    library/python/runtime
    library/python/symbols
)

IF (MUSL)
    PEERDIR(
        library/python/pythonapi
        library/python/musl
    )
ENDIF()

USE_PYTHON()

SRCS(
    main.c
)

END()

NEED_CHECK()
