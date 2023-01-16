PY23_LIBRARY()

LICENSE(ZPL-2.1)

VERSION(0.8.1)



PY_SRCS(
    TOP_LEVEL
    simplegeneric.py
)

NO_LINT()

RESOURCE_FILES(
    PREFIX contrib/python/simplegeneric/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
