PY23_LIBRARY()

LICENSE(
    MIT
)

VERSION(4.0.1)



PEERDIR(
    contrib/python/atomicwrites
    contrib/python/attrs
    contrib/python/more-itertools
    contrib/python/pluggy
    contrib/python/py
    contrib/python/setuptools
    contrib/python/six
)

IF (PYTHON2)
    PEERDIR(
        contrib/python/funcsigs
        contrib/python/pathlib2
    )
ENDIF()

NO_CHECK_IMPORTS(
    __tests__.*  # all test modules get imported when tests are run
    _pytest.*
)

PY_SRCS(
    TOP_LEVEL
    _pytest/_argcomplete.py
    _pytest/assertion/__init__.py
    _pytest/assertion/rewrite.py
    _pytest/assertion/truncate.py
    _pytest/assertion/util.py
    _pytest/cacheprovider.py
    _pytest/capture.py
    _pytest/_code/code.py
    _pytest/_code/__init__.py
    _pytest/_code/_py2traceback.py
    _pytest/_code/source.py
    _pytest/compat.py
    _pytest/config/argparsing.py
    _pytest/config/exceptions.py
    _pytest/config/findpaths.py
    _pytest/config/__init__.py
    _pytest/debugging.py
    _pytest/deprecated.py
    _pytest/doctest.py
    _pytest/fixtures.py
    _pytest/freeze_support.py
    _pytest/helpconfig.py
    _pytest/hookspec.py
    _pytest/__init__.py
    _pytest/junitxml.py
    _pytest/logging.py
    _pytest/main.py
    _pytest/mark/evaluate.py
    _pytest/mark/__init__.py
    _pytest/mark/legacy.py
    _pytest/mark/structures.py
    _pytest/monkeypatch.py
    _pytest/nodes.py
    _pytest/nose.py
    _pytest/outcomes.py
    _pytest/pastebin.py
    _pytest/pathlib.py
    _pytest/pytester.py
    _pytest/python_api.py
    _pytest/python.py
    _pytest/recwarn.py
    _pytest/reports.py
    _pytest/resultlog.py
    _pytest/runner.py
    _pytest/setuponly.py
    _pytest/setupplan.py
    _pytest/skipping.py
    _pytest/stepwise.py
    _pytest/terminal.py
    _pytest/tmpdir.py
    _pytest/unittest.py
    _pytest/_version.py
    _pytest/warnings.py
    _pytest/warning_types.py
    pytest.py
)

RESOURCE_FILES(
    PREFIX contrib/python/pytest/
    .dist-info/METADATA
    .dist-info/entry_points.txt
    .dist-info/top_level.txt
)

NO_LINT()

END()
