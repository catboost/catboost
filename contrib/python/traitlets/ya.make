PY23_LIBRARY()



IF (PYTHON2)
    PEERDIR(
        contrib/python/traitlets/py2
    )
ELSE()
    PEERDIR(
        contrib/python/traitlets/py3
    )
ENDIF()

NO_LINT()

END()

RECURSE(
    py2
    py3
)
