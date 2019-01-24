LIBRARY()



PEERDIR(
    library/python/runtime_py3
    library/python/symbols/module
    library/python/symbols/libc
    library/python/symbols/uuid
)

IF (MUSL)
    PEERDIR(
        library/python/pythonapi
    )
ENDIF()

USE_PYTHON3()

SRCS(
    main.c
)

END()
