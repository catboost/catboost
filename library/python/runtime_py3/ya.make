PY3_LIBRARY()



NO_WSHADOW()

PEERDIR(
    contrib/tools/python3/src
    contrib/tools/python3/lib/py
    library/cpp/resource
)

CFLAGS(-DCYTHON_REGISTER_ABCS=0)

NO_PYTHON_INCLUDES()

ENABLE(PYBUILD_NO_PYC)

PY_SRCS(
    entry_points.py
    TOP_LEVEL

    CYTHON_DIRECTIVE
    language_level=3

    __res.pyx
    sitecustomize.pyx
)

IF (CYTHON_COVERAGE)
    # Let covarage support add all needed files to resources
ELSE()
    RESOURCE_FILES(
        PREFIX ${MODDIR}/
        __res.pyx
        importer.pxi
        sitecustomize.pyx
    )
ENDIF()

END()

RECURSE_FOR_TESTS(
    test
)
