LIBRARY()

VERSION(3.0.7)



PEERDIR(
    contrib/python/py-1.4.30
    contrib/python/setuptools
)

PY_SRCS(
    TOP_LEVEL
    _pytest/runner.py
    _pytest/_code/__init__.py
    _pytest/_code/_py2traceback.py
    _pytest/_code/code.py
    _pytest/_code/source.py
    _pytest/assertion/__init__.py
    _pytest/assertion/rewrite.py
    _pytest/assertion/util.py
    _pytest/vendored_packages/__init__.py
    _pytest/vendored_packages/pluggy.py
    _pytest/__init__.py
    _pytest/_argcomplete.py
    _pytest/_pluggy.py
    _pytest/cacheprovider.py
    _pytest/capture.py
    _pytest/compat.py
    _pytest/config.py
    _pytest/debugging.py
    _pytest/deprecated.py
    _pytest/doctest.py
    _pytest/fixtures.py
    _pytest/freeze_support.py
    _pytest/helpconfig.py
    _pytest/hookspec.py
    _pytest/junitxml.py
    _pytest/main.py
    _pytest/mark.py
    _pytest/monkeypatch.py
    _pytest/nose.py
    _pytest/pastebin.py
    _pytest/pytester.py
    _pytest/python.py
    _pytest/recwarn.py
    _pytest/resultlog.py
    _pytest/setuponly.py
    _pytest/setupplan.py
    _pytest/skipping.py
    _pytest/terminal.py
    _pytest/tmpdir.py
    _pytest/unittest.py
    pytest.py
)

NO_LINT()

END()
