PY23_LIBRARY()



IF (PYTHON2)
    PEERDIR(
        contrib/python/pandas/py2
    )
ELSE()
    PEERDIR(
        contrib/python/pandas/py3
    )
ENDIF()

NO_LINT()

END()

RECURSE(
    py2
    py3
)
