PY23_LIBRARY()



IF (PYTHON2)
    PEERDIR(
        contrib/python/numpy/py2
    )
ELSE()
    PEERDIR(
        contrib/python/numpy/py2
    )
ENDIF()

NO_LINT()

END()

RECURSE(
    py2
    py3
)
