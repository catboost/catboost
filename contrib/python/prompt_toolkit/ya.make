PY23_LIBRARY()

# Proxy library
LICENSE(Not-Applicable)



IF (PYTHON2)
    PEERDIR(contrib/python/prompt_toolkit/py2)
ELSE()
    PEERDIR(contrib/python/prompt_toolkit/py3)
ENDIF()

NO_LINT()

END()

RECURSE(
    py2
    py3
)
