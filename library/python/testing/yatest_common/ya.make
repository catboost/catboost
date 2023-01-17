

PY23_LIBRARY()



NO_EXTENDED_SOURCE_SEARCH()

PY_SRCS(
    TOP_LEVEL
    yatest/__init__.py
    yatest/common/__init__.py
    yatest/common/benchmark.py
    yatest/common/canonical.py
    yatest/common/environment.py
    yatest/common/errors.py
    yatest/common/misc.py
    yatest/common/network.py
    yatest/common/path.py
    yatest/common/process.py
    yatest/common/runtime.py
    yatest/common/runtime_java.py
)

STYLE_PYTHON()

PEERDIR(
    contrib/python/packaging
    contrib/python/six
    library/python/cores
    library/python/filelock
    library/python/fs
    library/python/testing/yatest_lib
)

IF (NOT CATBOOST_OPENSOURCE)
    PEERDIR(
        library/python/coredump_filter
    )
ENDIF()

END()
