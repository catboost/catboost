PY23_LIBRARY()

LICENSE(BSD-3-Clause)



VERSION(0.13.8)

PEERDIR(
    contrib/python/decorator
    contrib/python/ipython
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    ipdb/__init__.py
    ipdb/__main__.py
    ipdb/stdout.py
)

NO_CHECK_IMPORTS(
    # Modules presented below leads to initialization of pdb,
    # which try to create ~/.ipython/profile_default/history.sqlite-journal,
    # due to which import tests may crash
    ipdb.__init__
    ipdb.__main__
    ipdb.stdout
)

RESOURCE_FILES(
    PREFIX contrib/python/ipdb/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
