import logging

import pytest

from library.python.pytest.plugins import ya


@pytest.fixture
def root_logger():
    """
    setup_logging reconfigures the root logger, which is also used by the running test itself.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level

    yield root

    for handler in root.handlers[:]:
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def ya_handlers(root):
    return [h for h in root.handlers if isinstance(h, ya.YaTestLoggingFileHandler)]


def test_handler_is_added_for_every_log(root_logger, tmpdir):
    common_log, test_log = str(tmpdir.join("run.log")), str(tmpdir.join("test.log"))

    ya.setup_logging(common_log, logging.DEBUG, test_log)

    assert sorted(h.baseFilename for h in ya_handlers(root_logger)) == sorted([common_log, test_log])


def test_handler_of_unchanged_log_is_reused(root_logger, tmpdir):
    common_log = str(tmpdir.join("run.log"))

    ya.setup_logging(common_log, logging.DEBUG, str(tmpdir.join("first.log")))
    reused = [h for h in ya_handlers(root_logger) if h.baseFilename == common_log]

    ya.setup_logging(common_log, logging.DEBUG, str(tmpdir.join("second.log")))

    assert [h for h in ya_handlers(root_logger) if h.baseFilename == common_log] == reused


def test_handler_of_dropped_log_is_removed(root_logger, tmpdir):
    common_log, first_log = str(tmpdir.join("run.log")), str(tmpdir.join("first.log"))

    ya.setup_logging(common_log, logging.DEBUG, first_log)
    ya.setup_logging(common_log, logging.DEBUG, str(tmpdir.join("second.log")))

    assert first_log not in [h.baseFilename for h in ya_handlers(root_logger)]


def test_every_log_receives_the_message(root_logger, tmpdir):
    common_log, test_log = str(tmpdir.join("run.log")), str(tmpdir.join("test.log"))
    ya.setup_logging(common_log, logging.DEBUG, test_log)

    logging.getLogger("ya.test").info("message to log")

    for log in (common_log, test_log):
        with open(log) as afile:
            assert "message to log" in afile.read()


def test_token_from_environment_is_masked(root_logger, tmpdir, monkeypatch):
    monkeypatch.setenv("SOME_SECRET_TOKEN", "secret-value")
    common_log = str(tmpdir.join("run.log"))
    ya.setup_logging(common_log, logging.DEBUG)

    logging.getLogger("ya.test").info("got secret-value from environment")

    with open(common_log) as afile:
        content = afile.read()
    assert "secret-value" not in content
    assert "[SECRET]" in content


def test_token_appeared_during_the_run_is_masked(root_logger, tmpdir, monkeypatch):
    common_log = str(tmpdir.join("run.log"))
    ya.setup_logging(common_log, logging.DEBUG, str(tmpdir.join("first.log")))

    # a test or a fixture may put a token into the environment while the suite is running
    monkeypatch.setenv("LATE_SECRET_TOKEN", "late-secret")
    ya.setup_logging(common_log, logging.DEBUG, str(tmpdir.join("second.log")))

    logging.getLogger("ya.test").info("got late-secret from environment")

    with open(common_log) as afile:
        content = afile.read()
    assert "late-secret" not in content
