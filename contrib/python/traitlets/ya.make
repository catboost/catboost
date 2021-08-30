PY23_LIBRARY()

# Proxy library
LICENSE(Not-Applicable)



IF (PYTHON2)
    PEERDIR(contrib/python/traitlets/py2)
ELSE()
    PEERDIR(contrib/python/traitlets/py3)
ENDIF()

NO_LINT()

END()

RECURSE(
    py2
    py3
)
