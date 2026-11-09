import pytest
import six
from _pytest.assertion import util

import library.python.pytest.rewrite as rewrite


class _Importer(object):
    def __init__(self, source):
        self.source = source

    def get_source(self, name):
        return self.source

    def get_filename(self, name):
        return "fake/test_module.py"


class _PassHookConfig(object):
    def __init__(self, config):
        self._assertstate = rewrite._get_state(config)

    def getini(self, name):
        return name == "enable_assertion_pass_hook"


def _rewrite(monkeypatch, config, source):
    monkeypatch.setattr(rewrite, "importer", _Importer(source))
    return rewrite._rewrite_test(config, "fake")


def test_tuple_assert_warns(request, monkeypatch):
    with pytest.warns(pytest.PytestAssertRewriteWarning, match="always true") as record:
        co = _rewrite(monkeypatch, request.config, "assert (1, 'msg')\n")
    assert co is not None
    assert record[0].lineno == 1
    if six.PY3:
        assert record[0].filename == "fake/test_module.py"


@pytest.mark.skipif(six.PY2, reason="pytest 4.6 has no pytest_assertion_pass")
def test_assertion_pass_hook(request, monkeypatch):
    calls = []
    monkeypatch.setattr(util, "_assertion_pass", lambda lineno, orig, expl: calls.append((lineno, orig)))
    co = _rewrite(monkeypatch, _PassHookConfig(request.config), u'x = u"\u0436"\nassert x == u"\u0436"\n')
    exec(co, {})
    assert calls == [(2, u'x == u"\u0436"')]
