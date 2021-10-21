PY3_LIBRARY()



VERSION(3.0.2)

LICENSE(BSD-3-Clause)

PEERDIR(
    contrib/python/MarkupSafe
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    jinja2/__init__.py
    jinja2/_identifier.py
    jinja2/async_utils.py
    jinja2/bccache.py
    jinja2/compiler.py
    jinja2/constants.py
    jinja2/debug.py
    jinja2/defaults.py
    jinja2/environment.py
    jinja2/exceptions.py
    jinja2/ext.py
    jinja2/filters.py
    jinja2/idtracking.py
    jinja2/lexer.py
    jinja2/loaders.py
    jinja2/meta.py
    jinja2/nativetypes.py
    jinja2/nodes.py
    jinja2/optimizer.py
    jinja2/parser.py
    jinja2/runtime.py
    jinja2/sandbox.py
    jinja2/tests.py
    jinja2/utils.py
    jinja2/visitor.py
)

RESOURCE_FILES(
    PREFIX contrib/python/Jinja2/py3/
    .dist-info/METADATA
    .dist-info/entry_points.txt
    .dist-info/top_level.txt
    jinja2/py.typed
)

END()

RECURSE_FOR_TESTS(
    tests
)
