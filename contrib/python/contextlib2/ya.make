PY23_LIBRARY() # Backport from Python 3.

LICENSE(PSF-2.0)



VERSION(0.6.0.post1)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    contextlib2.py
)

RESOURCE_FILES(
    PREFIX contrib/python/contextlib2/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
