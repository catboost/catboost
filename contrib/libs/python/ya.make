

PY23_LIBRARY()

NO_PYTHON_INCLUDES()

IF (USE_ARCADIA_PYTHON)
    ADDINCL(GLOBAL contrib/libs/python/Include)
    IF (MODULE_TAG STREQUAL "PY2")
        CFLAGS(GLOBAL -DUSE_PYTHON2)
        PEERDIR(contrib/tools/python/lib)
    ELSE()
        CFLAGS(GLOBAL -DUSE_PYTHON3)
        PEERDIR(contrib/tools/python3/lib)
    ENDIF()
ELSE()
    IF (USE_SYSTEM_PYTHON)
        PEERDIR(contrib/libs/platform/python)
    ELSE()
        CFLAGS(GLOBAL $PYTHON_INCLUDE)
    ENDIF()
ENDIF()
END()