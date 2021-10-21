PY23_LIBRARY()

# Proxy library
LICENSE(Not-Applicable)



IF (PYTHON2)
    PEERDIR(contrib/python/Jinja2/py2)
ELSE()
    PEERDIR(contrib/python/Jinja2/py3)
ENDIF()

NO_LINT()

END()

RECURSE(
    examples
    py2
    py3
)
