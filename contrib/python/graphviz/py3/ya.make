PY3_LIBRARY()

LICENSE(MIT)



VERSION(0.17)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    graphviz/__init__.py
    graphviz/backend.py
    graphviz/dot.py
    graphviz/files.py
    graphviz/lang.py
    graphviz/tools.py
)

RESOURCE_FILES(
    PREFIX contrib/python/graphviz/py3/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
