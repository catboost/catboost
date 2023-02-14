import os
import sys

from exts import func


@func.lazy
def is_tty():
    enforce_tty = os.environ.get('ENFORCE_TTY') is not None
    enforce_notty = os.environ.get('ENFORCE_NOTTY') is not None
    assert not (enforce_tty and enforce_notty)
    return not enforce_notty and (enforce_tty or sys.stderr.isatty())


class change_dir(object):
    """
    Saves and changes current directory to the specified one. Restores it on __exit__()
    """

    def __init__(self, *path):
        self.current_dir = os.getcwd()
        if path:
            os.chdir(os.path.join(*path))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        os.chdir(self.current_dir)


try:
    import scandir

    fastwalk = scandir.walk
except ImportError:
    fastwalk = os.walk
