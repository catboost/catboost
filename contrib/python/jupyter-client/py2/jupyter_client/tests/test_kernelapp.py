from __future__ import division

import os
import shutil
from subprocess import Popen, PIPE
import sys
from tempfile import mkdtemp
import time

from .utils import _JUPYTER_KERNEL_BIN

PY3 = sys.version_info[0] >= 3

def _launch(extra_env):
    env = os.environ.copy()
    env.update(extra_env)
    return Popen([_JUPYTER_KERNEL_BIN()], env=env, stderr=(PIPE if PY3 else None))

WAIT_TIME = 10
POLL_FREQ = 10

def hacky_wait(p):
    """Python 2 subprocess doesn't have timeouts :-("""
    for _ in range(WAIT_TIME * POLL_FREQ):
        if p.poll() is not None:
            return p.returncode
        time.sleep(1 / POLL_FREQ)
    else:
        raise AssertionError("Process didn't exit in {} seconds"
                             .format(WAIT_TIME))

def test_kernelapp_lifecycle():
    # Check that 'jupyter kernel' starts and terminates OK.
    runtime_dir = mkdtemp()
    startup_dir = mkdtemp()
    started = os.path.join(startup_dir, 'started')
    try:
        p = _launch({'JUPYTER_RUNTIME_DIR': runtime_dir,
                     'JUPYTER_CLIENT_TEST_RECORD_STARTUP_PRIVATE': started,
                    })
        # Wait for start
        for _ in range(WAIT_TIME * POLL_FREQ):
            if os.path.isfile(started):
                break
            time.sleep(1 / POLL_FREQ)
        else:
            raise AssertionError("No started file created in {} seconds"
                                 .format(WAIT_TIME))

        # Connection file should be there by now
        files = os.listdir(runtime_dir)
        assert len(files) == 1
        cf = files[0]
        assert cf.startswith('kernel')
        assert cf.endswith('.json')

        # Send SIGTERM to shut down
        p.terminate()
        if PY3:
            _, stderr = p.communicate(timeout=WAIT_TIME)
            assert cf in stderr.decode('utf-8', 'replace')
        else:
            hacky_wait(p)
    finally:
        shutil.rmtree(runtime_dir)
        shutil.rmtree(startup_dir)

