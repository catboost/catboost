PY23_LIBRARY()

LICENSE(
    MIT
)

VERSION(2.3.5)



PY_SRCS(
    TOP_LEVEL
    pathlib2/__init__.py
)

PEERDIR(contrib/python/six)

IF (PYTHON2)
    PEERDIR(
        contrib/python/scandir
    )
ENDIF()

RESOURCE_FILES(
    PREFIX contrib/python/pathlib2/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

NO_LINT()

END()
