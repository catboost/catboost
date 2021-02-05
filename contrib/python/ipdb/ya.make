PY23_LIBRARY()

LICENSE(BSD-3-Clause)



VERSION(0.13.4)

PEERDIR(
    contrib/python/ipython
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    ipdb/__init__.py
    ipdb/__main__.py
    ipdb/stdout.py
)

RESOURCE_FILES(
    PREFIX contrib/python/ipdb/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
