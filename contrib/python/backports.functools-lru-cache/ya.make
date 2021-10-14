PY2_LIBRARY()



LICENSE(MIT)

VERSION(1.6.1)

PY_SRCS(
    NAMESPACE backports
    functools_lru_cache.py
)

NO_LINT()

RESOURCE_FILES(
    PREFIX contrib/python/backports.functools-lru-cache/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
