PY23_LIBRARY()

LICENSE(BSD-3-Clause)



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
