import contextlib
import os

@contextlib.contextmanager
def temporary_env(newenv):
    """Completely replace the environment variables with the specified dict.
    
    Use as a context manager::
    
        with temporary_env({'PATH': my_path}):
            ...
    """
    orig_env = os.environ.copy()
    os.environ.clear()
    os.environ.update(newenv)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(orig_env)    

@contextlib.contextmanager
def modified_env(changes, snapshot=True):
    """Temporarily modify environment variables.
    
    Specify the changes as a dictionary mapping names to new values, using
    None as the value for names that should be deleted.
    
    Example use::
    
        with modified_env({'SHELL': 'bash', 'PYTHONPATH': None}):
            ...
    
    When the context exits, there are two possible ways to restore the
    environment. If *snapshot* is True, the default, it will reset the whole
    environment to its state when the context was entered. If *snapshot* is
    False, it will restore only the specific variables it modified, leaving
    any changes made to other environment variables in the context.
    """
    def update_del(changes):
        for k, v in changes.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    
    if snapshot:
        saved_variables = os.environ.copy()
    else:
        saved_variables = {}
        for k,v in changes.items():
            saved_variables[k] = os.environ.get(k, None)

    update_del(changes)
    
    try:
        yield
    finally:
        if snapshot:
            os.environ.clear()
            os.environ.update(saved_variables)
        else:
            update_del(saved_variables)

def make_env_restorer():
    """Snapshot the current environment, return a function to restore that.

    This is intended to produce cleanup functions for tests. For example,
    using the :class:`unittest.TestCase` API::

        def setUp(self):
            self.addCleanup(testpath.make_env_restorer())

    Any changes a test makes to the environment variables will be wiped out
    before the next test is run.
    """
    orig_env = os.environ.copy()

    def restore():
        os.environ.clear()
        os.environ.update(orig_env)

    return restore
