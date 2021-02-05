PY23_LIBRARY()

LICENSE(MIT)



VERSION(10.3.1)

RESOURCE_FILES(
    PREFIX contrib/python/path.py/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    path.py
)

END()
