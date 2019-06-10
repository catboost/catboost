import subprocess  # noqa: F401

import pytest

from pandas.io.formats.console import detect_console_encoding
from pandas.io.formats.terminal import _get_terminal_size_tput


class MockEncoding(object):  # TODO(py27): replace with mock
    """
    Used to add a side effect when accessing the 'encoding' property. If the
    side effect is a str in nature, the value will be returned. Otherwise, the
    side effect should be an exception that will be raised.
    """
    def __init__(self, encoding):
        super(MockEncoding, self).__init__()
        self.val = encoding

    @property
    def encoding(self):
        return self.raise_or_return(self.val)

    @staticmethod
    def raise_or_return(val):
        if isinstance(val, str):
            return val
        else:
            raise val


@pytest.mark.parametrize('empty,filled', [
    ['stdin', 'stdout'],
    ['stdout', 'stdin']
])
def test_detect_console_encoding_from_stdout_stdin(monkeypatch, empty, filled):
    # Ensures that when sys.stdout.encoding or sys.stdin.encoding is used when
    # they have values filled.
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr('sys.{}'.format(empty), MockEncoding(''))
        context.setattr('sys.{}'.format(filled), MockEncoding(filled))
        assert detect_console_encoding() == filled


@pytest.mark.parametrize('encoding', [
    AttributeError,
    IOError,
    'ascii'
])
def test_detect_console_encoding_fallback_to_locale(monkeypatch, encoding):
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr('locale.getpreferredencoding', lambda: 'foo')
        context.setattr('sys.stdout', MockEncoding(encoding))
        assert detect_console_encoding() == 'foo'


@pytest.mark.parametrize('std,locale', [
    ['ascii', 'ascii'],
    ['ascii', Exception],
    [AttributeError, 'ascii'],
    [AttributeError, Exception],
    [IOError, 'ascii'],
    [IOError, Exception]
])
def test_detect_console_encoding_fallback_to_default(monkeypatch, std, locale):
    # When both the stdout/stdin encoding and locale preferred encoding checks
    # fail (or return 'ascii', we should default to the sys default encoding.
    # GH 21552
    with monkeypatch.context() as context:
        context.setattr(
            'locale.getpreferredencoding',
            lambda: MockEncoding.raise_or_return(locale)
        )
        context.setattr('sys.stdout', MockEncoding(std))
        context.setattr('sys.getdefaultencoding', lambda: 'sysDefaultEncoding')
        assert detect_console_encoding() == 'sysDefaultEncoding'


@pytest.mark.parametrize("size", ['', ['']])
def test_terminal_unknown_dimensions(monkeypatch, size, mocker):

    def communicate(*args, **kwargs):
        return size

    monkeypatch.setattr('subprocess.Popen', mocker.Mock())
    monkeypatch.setattr('subprocess.Popen.return_value.returncode', None)
    monkeypatch.setattr(
        'subprocess.Popen.return_value.communicate', communicate)
    result = _get_terminal_size_tput()

    assert result is None
