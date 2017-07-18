LIBRARY()



PEERDIR(
    library/python/runtime
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
