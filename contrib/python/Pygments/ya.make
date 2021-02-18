PY23_LIBRARY()

LICENSE(BSD-3-Clause)



IF (PYTHON2)
    PEERDIR(
        contrib/python/Pygments/py2
    )
ELSE()
    PEERDIR(
        contrib/python/Pygments/py3
    )
ENDIF()

NO_LINT()

END()

RECURSE(
    py2
    py3
)
