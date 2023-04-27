"""Testing utils for jupyter_client tests

"""
import os
pjoin = os.path.join
import sys
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

import pytest

from ipython_genutils.tempdir import TemporaryDirectory

PY3 = sys.version_info[0] >= 3

JUPYTER_KERNEL_BIN_PATH = "contrib/python/jupyter-client/py2/bin/jupyter_kernel/{}/jupyter-kernel"
JUPYTER_KERNEL_BIN_PATH = JUPYTER_KERNEL_BIN_PATH.format('py3' if PY3 else 'py2')

def _JUPYTER_KERNEL_BIN():
    import yatest
    return yatest.common.binary_path(JUPYTER_KERNEL_BIN_PATH)
JUPYTER_KERNELSPEC_BIN_PATH = "contrib/python/jupyter-client/py2/bin/jupyter_kernelspec/{}/jupyter-kernelspec"
JUPYTER_KERNELSPEC_BIN_PATH = JUPYTER_KERNELSPEC_BIN_PATH.format('py3' if PY3 else 'py2')
def _JUPYTER_KERNELSPEC_BIN():
    import yatest
    return yatest.common.binary_path(JUPYTER_KERNELSPEC_BIN_PATH)

skip_win32 = pytest.mark.skipif(sys.platform.startswith('win'), reason="Windows")


class test_env(object):
    """Set Jupyter path variables to a temporary directory

    Useful as a context manager or with explicit start/stop
    """
    def start(self):
        self.test_dir = td = TemporaryDirectory()
        self.env_patch = patch.dict(os.environ, {
            'JUPYTER_CONFIG_DIR': pjoin(td.name, 'jupyter'),
            'JUPYTER_DATA_DIR': pjoin(td.name, 'jupyter_data'),
            'JUPYTER_RUNTIME_DIR': pjoin(td.name, 'jupyter_runtime'),
            'IPYTHONDIR': pjoin(td.name, 'ipython'),
        })
        self.env_patch.start()
    
    def stop(self):
        self.env_patch.stop()
        self.test_dir.cleanup()
    
    def __enter__(self):
        self.start()
        return self.test_dir.name
    
    def __exit__(self, *exc_info):
        self.stop()

def execute(code='', kc=None, **kwargs):
    """wrapper for doing common steps for validating an execution request"""
    from .test_message_spec import validate_message
    if kc is None:
        kc = KC
    msg_id = kc.execute(code=code, **kwargs)
    reply = kc.get_shell_msg(timeout=TIMEOUT)
    validate_message(reply, 'execute_reply', msg_id)
    busy = kc.get_iopub_msg(timeout=TIMEOUT)
    validate_message(busy, 'status', msg_id)
    assert busy['content']['execution_state'] == 'busy'

    if not kwargs.get('silent'):
        execute_input = kc.get_iopub_msg(timeout=TIMEOUT)
        validate_message(execute_input, 'execute_input', msg_id)
        assert execute_input['content']['code'] == code

    return msg_id, reply['content']
