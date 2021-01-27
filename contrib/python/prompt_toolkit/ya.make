PY23_LIBRARY()



IF (PYTHON2)
    PEERDIR(
        contrib/python/prompt_toolkit/py2
    )
ELSE()
    PEERDIR(
        contrib/python/prompt_toolkit/py3
    )
ENDIF()

NO_LINT()

END()

RECURSE(
    py2
)
