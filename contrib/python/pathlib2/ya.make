PY23_LIBRARY()

LICENSE(
    MIT
)

VERSION(2.3.3)



PY_SRCS(
    TOP_LEVEL
    pathlib2/__init__.py
)

PEERDIR(contrib/python/six)

IF (PYTHON2)
    PEERDIR(
        contrib/python/scandir
    )
ENDIF()

NO_LINT()

END()
