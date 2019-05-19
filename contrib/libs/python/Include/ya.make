PY23_LIBRARY()



NO_PYTHON_INCLUDES()

ADDINCL(GLOBAL contrib/libs/python/Include)

IF (PYTHON2)
    CFLAGS(GLOBAL -DUSE_PYTHON2)

    PEERDIR(contrib/tools/python/lib)
ELSE()
    CFLAGS(GLOBAL -DUSE_PYTHON3)

    PEERDIR(contrib/tools/python3/src)
ENDIF()

END()
