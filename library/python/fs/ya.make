

PY23_LIBRARY()

PY_SRCS(
    __init__.py
)

IF (OS_DARWIN)
    PY_SRCS(
        clonefile.pyx
    )
ENDIF()

PEERDIR(
    library/python/func
    library/python/strings
    library/python/windows
)

END()

RECURSE_FOR_TESTS(test)
