

PY23_LIBRARY()

NO_PYTHON_INCLUDES()

IF (USE_ARCADIA_PYTHON)
    PEERDIR(
        contrib/libs/python/Include
        library/python/symbols/module
        library/python/symbols/libc
        library/python/symbols/python
    )

    IF (NOT OS_WINDOWS)
        PEERDIR(
        )
    ENDIF()

    IF (MODULE_TAG STREQUAL "PY2")
        PEERDIR(
            contrib/tools/python/lib
            library/python/runtime
        )
    ELSE()
        PEERDIR(
            contrib/tools/python3/lib
            library/python/runtime_py3
        )
    ENDIF()
ELSE()
    IF (USE_SYSTEM_PYTHON)
        PEERDIR(build/platform/python)
    ELSE()
        CFLAGS(GLOBAL $PYTHON_INCLUDE)
    ENDIF()
ENDIF()
END()

RECURSE(
    Include
)
