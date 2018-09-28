LIBRARY()



PEERDIR(
    library/python/runtime_py3
)

#IF (MUSL)
#    PEERDIR(
#        library/python/pythonapi
#    )
#ENDIF()

USE_PYTHON3()

SRCS(
    main.c
)

END()

NEED_CHECK()
