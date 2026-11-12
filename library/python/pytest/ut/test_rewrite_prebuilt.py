import pytest
import six

from library.python.pytest import rewrite


@pytest.mark.skipif(not six.PY3, reason='build-time rewrite is py3 only')
def test_rewrite_test_takes_prebuilt(request, monkeypatch):
    monkeypatch.setattr(rewrite.rewrite, 'rewrite_asserts', lambda *args, **kwargs: pytest.fail('rewritten at import'))
    assert rewrite._rewrite_test(request.config, __name__) is not None


@pytest.mark.skipif(not six.PY3, reason='build-time rewrite is py3 only')
def test_module_with_rewrite_warning_has_no_prebuilt(request):
    name = __name__.rsplit('.', 1)[0] + '.rewrite_warning'
    assert rewrite.importer.get_source(name)
    assert rewrite._prebuilt_code(request.config, name) is None


class _PassHookConfig(object):
    def getini(self, name):
        return name == 'enable_assertion_pass_hook'


@pytest.mark.skipif(not six.PY3, reason='build-time rewrite is py3 only')
def test_no_prebuilt_with_assertion_pass_hook():
    assert rewrite._prebuilt_code(_PassHookConfig(), __name__) is None


@pytest.mark.skipif(not six.PY3, reason='build-time rewrite is py3 only')
def test_no_prebuilt_for_source_on_disk(request, monkeypatch):
    filename = rewrite.importer.get_filename(__name__)
    resolve = rewrite.__res.resfs_resolve
    monkeypatch.setattr(rewrite.__res, 'resfs_resolve', lambda path: path if path == filename else resolve(path))
    assert rewrite._prebuilt_code(request.config, __name__) is None


@pytest.mark.skipif(not six.PY3, reason='build-time rewrite is py3 only')
def test_no_prebuilt_on_header_mismatch(request, monkeypatch):
    monkeypatch.setattr(rewrite, '_PREBUILT_HEADER', b'pytest-other\n')
    assert rewrite._prebuilt_code(request.config, __name__) is None


def test_assert_is_rewritten():
    x = 1
    with pytest.raises(AssertionError) as e:
        assert x == 2
    assert '1 == 2' in str(e.value)


def _filenames(co):
    res = [co.co_filename]
    for c in co.co_consts:
        if isinstance(c, type(co)):
            res += _filenames(c)
    return res


@pytest.mark.skipif(not six.PY3, reason='build-time rewrite is py3 only')
def test_prebuilt_code_matches_runtime_rewrite(request, monkeypatch):
    prebuilt = rewrite._prebuilt_code(request.config, __name__)
    monkeypatch.setattr(rewrite, '_prebuilt_code', lambda config, name: None)
    runtime = rewrite._rewrite_test(request.config, __name__)
    assert prebuilt == runtime
    assert _filenames(prebuilt) == _filenames(runtime)
