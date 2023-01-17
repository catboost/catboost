# NOTE: please do not change to PY23_LIBRARY()
# instead, use
# IF (PYTHON2)
#     PEERDIR(contrib/deprecated/python/typing)
# ENDIF()
# for code compatible with both Py2 and Py3
PY2_LIBRARY()  # backport

LICENSE(PSF-2.0)



VERSION(3.10.0.0)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    typing.py
)

RESOURCE_FILES(
    PREFIX contrib/deprecated/python/typing/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()

RECURSE_FOR_TESTS(
    test
)
