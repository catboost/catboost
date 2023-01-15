PY3_LIBRARY()

LICENSE(BSD-3-Clause)

VERSION(5.0.5)

PROVIDES(python_traitlets)



PEERDIR(
    contrib/python/ipython_genutils
)

PY_SRCS(
    TOP_LEVEL
    traitlets/__init__.py
    traitlets/_version.py
    traitlets/config/__init__.py
    traitlets/config/application.py
    traitlets/config/configurable.py
    traitlets/config/loader.py
    traitlets/config/manager.py
    traitlets/config/sphinxdoc.py
    traitlets/log.py
    traitlets/tests/__init__.py
    traitlets/tests/_warnings.py
    traitlets/tests/utils.py
    traitlets/traitlets.py
    traitlets/utils/__init__.py
    traitlets/utils/bunch.py
    traitlets/utils/decorators.py
    traitlets/utils/descriptions.py
    traitlets/utils/getargspec.py
    traitlets/utils/importstring.py
    traitlets/utils/sentinel.py
)

RESOURCE_FILES(
    PREFIX contrib/python/traitlets/py3/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    tests
)
