PY_LIBRARY()



NO_WSHADOW()

PEERDIR(
    contrib/tools/python/lib
    library/cpp/resource
)

CFLAGS(-DCYTHON_REGISTER_ABCS=0)

NO_PYTHON_INCLUDES()

PY_SRCS(
    entry_points.py
    TOP_LEVEL
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
