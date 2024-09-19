import logging
import pytest
import random
import multiprocessing

logger = logging.getLogger(__name__)


@pytest.fixture
def tmpfile(tmp_path):
    filename = tmp_path / str(random.random())
    yield str(filename)
    try:
        filename.unlink(missing_ok=True)
    except PermissionError:
        pass


def pytest_sessionstart(session):
    # Force spawning the process so we don't accidently inherit locks.
    # I'm not a 100% certain this will work correctly unfortunately... there
    # is some potential for breaking tests
    multiprocessing.set_start_method('spawn')
