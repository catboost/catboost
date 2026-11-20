PY3_PROGRAM(cython)

WITHOUT_LICENSE_TEXTS()

# Keep compiler bootstrap independent of Python 2 runtime and import tests.
ENABLE(NO_PYTHON_INCLS)

DISABLE(PYTHON_SQLITE3)

NO_CHECK_IMPORTS()

NO_IMPORT_TRACING()

NO_PYTHON_COVERAGE()

VERSION(3.2.9)

LICENSE(
    Apache-2.0 AND
    MIT
)

PEERDIR(
    contrib/tools/cython
    library/python/runtime_py3
)

IF (NOT USE_PYTHON3_PREV)
    PEERDIR(
        contrib/tools/python3/Lib
    )
ENDIF()

# Coverage extraction loads library.python.resource from every binary.
# Pack its wrapper directly to keep bootstrap independent of the PY23 module.
PY_SRCS(
    ${ARCADIA_ROOT}/library/python/resource/__init__.py=library.python.resource.__init__
    TOP_LEVEL
    main.py
)

PY_MAIN(main)

INDUCED_DEPS(
    h+cpp
    ${ARCADIA_ROOT}/contrib/tools/cython/induced_deps.h
)

END()

RECURSE_FOR_TESTS(
    tests
)
