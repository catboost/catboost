import logging
import os
import subprocess
import sys
import threading

from exts.strings import stringize_deep
import exts.windows
import six


class TimeoutExpired(Exception):
    def __init__(self, stdout='', stderr=''):
        self.stdout = stdout
        self.stderr = stderr


logger = logging.getLogger(__name__)


# Wrapper for Popen with some fixes and improvements
@exts.windows.errorfix
def popen(*args, **kwargs):
    if exts.windows.on_win():
        exts.windows.disable_error_dialogs()
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = exts.windows.default_process_creation_flags()
    return subprocess.Popen(*args, **kwargs)


# function below uses special code for win instead of os.execve because of: https://bugs.python.org/issue9148
def execve(exec_path, args=[], env=None, cwd=None):
    from exts import tmp

    tmp.remove_tmp_dirs(env)

    logging.shutdown()

    if env is None:
        env = os.environ

    if not exts.windows.on_win():
        if cwd:
            os.chdir(cwd)
        os.execve(exec_path, [exec_path] + args, env)  # after this call no return in current process
        assert False, 'WTF: no return from os.execve: {}.'.format([exec_path] + args)

    if env:
        env = stringize_deep(env)
    spr = popen([exec_path] + args, env=env, cwd=cwd)  # emulate exec for win
    spr.wait()
    sys.exit(spr.returncode)


def run_process(exec_path, args=[], env=None, cwd=None, check=False, pipe_stdout=True, return_stderr=False):
    logger.debug("run {0} with args {1} and {2} env".format(exec_path, args, env))
    process = popen(
        [exec_path] + args, stdout=subprocess.PIPE if pipe_stdout else None, stderr=subprocess.PIPE, env=env, cwd=cwd
    )
    output, errors = process.communicate()
    if check and process.returncode:
        logger.error(errors)
        raise subprocess.CalledProcessError(process.returncode, [exec_path] + args, output=output)
    if isinstance(output, bytes):
        output = output.decode("ascii")
    return output if not return_stderr else (output, errors)


# Legacy, use exts.process.popen wrapper
def subprocess_flags():
    if exts.windows.on_win():
        exts.windows.disable_error_dialogs()
        return exts.windows.default_process_creation_flags()
    return 0


def set_close_on_exec(stream):
    if exts.windows.on_win():
        exts.windows.set_handle_information(stream, inherit=False)
    else:
        import fcntl

        flags = fcntl.fcntl(stream, fcntl.F_GETFD)
        flags |= fcntl.FD_CLOEXEC
        fcntl.fcntl(stream, fcntl.F_SETFD, flags)


def wait_for_proc(proc, timeout=None):
    if timeout is None:
        return proc.communicate()

    res, err = [], []

    def run():
        try:
            res.extend(proc.communicate())
        except Exception:
            err.extend(sys.exc_info())

    th = threading.Thread(target=run)
    th.daemon = True
    th.start()

    th.join(timeout)
    alive = th.is_alive()
    proc.terminate()
    th.join()

    if err:
        six.reraise(err[0], err[1], err[2])
    if alive:
        raise TimeoutExpired(res[0], res[1])
    return res[0], res[1]
