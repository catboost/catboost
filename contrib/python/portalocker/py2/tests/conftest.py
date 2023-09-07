import py
import pytest


@pytest.fixture
def tmpfile(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('temp')
    filename = tmpdir.join('tmpfile')
    yield str(filename)
    try:
        filename.remove(ignore_errors=True)
    except (py.error.EBUSY, py.error.ENOENT):
        pass

