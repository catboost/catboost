PY23_LIBRARY(Jinja2)



VERSION(2.10.1)

PEERDIR(
    contrib/python/MarkupSafe
    contrib/python/setuptools
)

PY_SRCS(
    TOP_LEVEL
    jinja2/__init__.py
    jinja2/_compat.py
    jinja2/_identifier.py
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

IF (PYTHON3)
    PY_SRCS(
        TOP_LEVEL
        jinja2/asyncfilters.py
        jinja2/asyncsupport.py
    )
ENDIF()

NO_LINT()

END()

RECURSE_FOR_TESTS(
    tests
)
