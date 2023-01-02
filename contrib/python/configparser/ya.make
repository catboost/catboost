PY2_LIBRARY() # Backport from Python 3.

LICENSE(MIT)

VERSION(4.0.2)



NO_LINT()

PY_SRCS(
    TOP_LEVEL
    backports/configparser/helpers.py
    backports/configparser/__init__.py
    configparser.py
)

RESOURCE_FILES(
    PREFIX contrib/python/configparser/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
