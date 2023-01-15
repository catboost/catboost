PY23_LIBRARY()

LICENSE(BSD-3-Clause)



IF (PYTHON2)
    PEERDIR(
        contrib/python/ipython/py2
    )
ELSE()
    PEERDIR(
        contrib/python/ipython/py3
    )
ENDIF()

END()

RECURSE(
    bin
    py2
    py3
)
