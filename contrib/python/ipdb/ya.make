PY23_LIBRARY()



VERSION(0.13.4)

LICENSE(BSD-3-Clause)

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
