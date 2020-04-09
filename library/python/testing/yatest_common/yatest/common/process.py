# coding: utf-8

import os
import re
import time
import signal
import shutil
import logging
import tempfile
import subprocess
import errno
import distutils.version

import six

try:
    # yatest.common should try to be hermetic, otherwise, PYTEST_SCRIPT (aka USE_ARCADIA_PYTHON=no) won't work.
    import library.python.cores as cores
except ImportError:
    cores = None

from . import runtime
from . import path
from . import environment


MAX_OUT_LEN = 1000 * 1000  # 1 mb
MAX_MESSAGE_LEN = 1500
SANITIZER_ERROR_PATTERN = br": ([A-Z][\w]+Sanitizer)"
GLIBC_PATTERN = re.compile(r"\S+@GLIBC_([0-9.]+)")
yatest_logger = logging.getLogger("ya.test")


def truncate(s, size):
    if s is None:
        return None
    elif len(s) <= size:
        return s
    else:
        return (b'...' if isinstance(s, bytes) else '...') + s[-(size - 3):]


def get_command_name(command):
    return os.path.basename(command.split()[0] if isinstance(command, six.string_types) else command[0])


class ExecutionError(Exception):

    def __init__(self, execution_result):
        if not isinstance(execution_result.command, six.string_types):
            command = " ".join(str(arg) for arg in execution_result.command)
        else:
            command = execution_result.command
        message = "Command '{command}' has failed with code {code}.\nErrors:\n{err}\n".format(
            command=command,
            code=execution_result.exit_code,
            err=_format_error(execution_result.std_err))
        if cores:
            if execution_result.backtrace:
                message += "Backtrace:\n[[rst]]{}[[bad]]\n".format(cores.colorize_backtrace(execution_result._backtrace))
        else:
            message += "Backtrace is not available: module cores isn't available"

        super(ExecutionError, self).__init__(message)
        self.execution_result = execution_result


class TimeoutError(Exception):
    pass


class ExecutionTimeoutError(TimeoutError):
    def __init__(self, execution_result, *args, **kwargs):
        super(ExecutionTimeoutError, self).__init__(args, kwargs)
        self.execution_result = execution_result


class InvalidExecutionStateError(Exception):
    pass


class SignalInterruptionError(Exception):
    pass


class _Execution(object):

    def __init__(self, command, process, out_file, err_file, process_progress_listener=None, cwd=None, collect_cores=True, check_sanitizer=True, started=0, user_stdout=False, user_stderr=False):
        self._command = command
        self._process = process
        self._out_file = out_file
        self._err_file = err_file
        self._std_out = None
        self._std_err = None
        self._elapsed = None
        self._start = time.time()
        self._process_progress_listener = process_progress_listener
        self._cwd = cwd or os.getcwd()
        self._collect_cores = collect_cores
        self._backtrace = ''
        self._check_sanitizer = check_sanitizer
        self._metrics = {}
        self._started = started
        self._user_stdout = bool(user_stdout)
        self._user_stderr = bool(user_stderr)
        self._exit_code = None
        if process_progress_listener:
            process_progress_listener.open(command, process, out_file, err_file)

    @property
    def running(self):
        return self._process.poll() is None

    def kill(self):
        if self.running:
            self._save_outputs(False)
            _kill_process_tree(self._process.pid)
            self._clean_files()
            # DEVTOOLS-2347
            yatest_logger.debug("Process status before wait_for: %s", self.running)
            try:
                wait_for(lambda: not self.running, timeout=5, fail_message="Could not kill process {}".format(self._process.pid), sleep_time=.1)
            except TimeoutError:
                yatest_logger.debug("Process status after wait_for: %s", self.running)
                yatest_logger.debug("Process %d info: %s", self._process.pid, _get_proc_tree_info([self._process.pid]))
                raise
        else:
            raise InvalidExecutionStateError("Cannot kill a stopped process")

    def terminate(self):
        if self.running:
            self._process.terminate()

    @property
    def process(self):
        return self._process

    @property
    def command(self):
        return self._command

    @property
    def returncode(self):
        return self.exit_code

    @property
    def exit_code(self):
        """
        Deprecated, use returncode
        """
        if self._exit_code is None:
            self._exit_code = self._process.returncode
        return self._exit_code

    @property
    def stdout(self):
        return self.std_out

    @property
    def std_out(self):
        """
        Deprecated, use stdout
        """
        if self._std_out is not None:
            return self._std_out
        if self._process.stdout and not self._user_stdout:
            self._std_out = self._process.stdout.read()
            return self._std_out

    @property
    def stderr(self):
        return self.std_err

    @property
    def std_err(self):
        """
        Deprecated, use stderr
        """
        if self._std_err is not None:
            return self._std_err
        if self._process.stderr and not self._user_stderr:
            self._std_err = self._process.stderr.read()
            return self._std_err

    @property
    def elapsed(self):
        return self._elapsed

    @property
    def backtrace(self):
        return self._backtrace

    @property
    def metrics(self):
        return self._metrics

    def _save_outputs(self, clean_files=True):
        if self._process_progress_listener:
            self._process_progress_listener()
            self._process_progress_listener.close()
        if not self._user_stdout:
            if self._out_file is None:
                pass
            elif self._out_file != subprocess.PIPE:
                self._out_file.flush()
                self._out_file.seek(0, os.SEEK_SET)
                self._std_out = self._out_file.read()
            else:
                self._std_out = self._process.stdout.read()
        if not self._user_stderr:
            if self._err_file is None:
                pass
            elif self._err_file != subprocess.PIPE:
                self._err_file.flush()
                self._err_file.seek(0, os.SEEK_SET)
                self._std_err = self._err_file.read()
            else:
                self._std_err = self._process.stderr.read()

        if clean_files:
            self._clean_files()
        yatest_logger.debug("Command (pid %s) rc: %s", self._process.pid, self.exit_code)
        yatest_logger.debug("Command (pid %s) elapsed time (sec): %s", self._process.pid, self.elapsed)
        if self._metrics:
            for key, value in six.iteritems(self._metrics):
                yatest_logger.debug("Command (pid %s) %s: %s", self._process.pid, key, value)
        yatest_logger.debug("Command (pid %s) output:\n%s", self._process.pid, truncate(self._std_out, MAX_OUT_LEN))
        yatest_logger.debug("Command (pid %s) errors:\n%s", self._process.pid, truncate(self._std_err, MAX_OUT_LEN))

    def _clean_files(self):
        if self._err_file and not self._user_stderr and self._err_file != subprocess.PIPE:
            self._err_file.close()
            self._err_file = None
        if self._out_file and not self._user_stdout and self._out_file != subprocess.PIPE:
            self._out_file.close()
            self._out_file = None

    def _recover_core(self):
        core_path = cores.recover_core_dump_file(self.command[0], self._cwd, self.process.pid)
        if core_path:
            # Core dump file recovering may be disabled (for distbuild for example) - produce only bt
            store_cores = runtime._get_ya_config().collect_cores
            if store_cores:
                new_core_path = path.get_unique_file_path(runtime.output_path(), "{}.{}.core".format(os.path.basename(self.command[0]), self._process.pid))
                # Copy core dump file, because it may be overwritten
                yatest_logger.debug("Coping core dump file from '%s' to the '%s'", core_path, new_core_path)
                shutil.copyfile(core_path, new_core_path)
                core_path = new_core_path

            bt_filename = None
            pbt_filename = None

            if os.path.exists(runtime.gdb_path()):
                self._backtrace = cores.get_gdb_full_backtrace(self.command[0], core_path, runtime.gdb_path())
                bt_filename = path.get_unique_file_path(runtime.output_path(), "{}.{}.backtrace".format(os.path.basename(self.command[0]), self._process.pid))
                with open(bt_filename, "wb") as afile:
                    afile.write(self._backtrace)
                # generate pretty html version of backtrace aka Tri Korochki
                pbt_filename = bt_filename + ".html"
                backtrace_to_html(bt_filename, pbt_filename)

            if store_cores:
                runtime._register_core(os.path.basename(self.command[0]), self.command[0], core_path, bt_filename, pbt_filename)
            else:
                runtime._register_core(os.path.basename(self.command[0]), None, None, bt_filename, pbt_filename)

    def wait(self, check_exit_code=True, timeout=None, on_timeout=None):
        def _wait():
            finished = None
            interrupted = False
            try:
                if hasattr(os, "wait4"):
                    try:
                        if hasattr(subprocess, "_eintr_retry_call"):
                            pid, sts, rusage = subprocess._eintr_retry_call(os.wait4, self._process.pid, 0)
                        else:
                            # PEP 475
                            pid, sts, rusage = os.wait4(self._process.pid, 0)
                        finished = time.time()
                        self._process._handle_exitstatus(sts)
                        for field in [
                            "ru_idrss",
                            "ru_inblock",
                            "ru_isrss",
                            "ru_ixrss",
                            "ru_majflt",
                            "ru_maxrss",
                            "ru_minflt",
                            "ru_msgrcv",
                            "ru_msgsnd",
                            "ru_nivcsw",
                            "ru_nsignals",
                            "ru_nswap",
                            "ru_nvcsw",
                            "ru_oublock",
                            "ru_stime",
                            "ru_utime",
                        ]:
                            if hasattr(rusage, field):
                                self._metrics[field.replace("ru_", "")] = getattr(rusage, field)
                    except OSError as exc:

                        if exc.errno == errno.ECHILD:
                            yatest_logger.debug("Process resource usage is not available as process finished before wait4 was called")
                        else:
                            raise
            except SignalInterruptionError:
                interrupted = True
                raise
            finally:
                if not interrupted:
                    self._process.wait()  # this has to be here unconditionally, so that all process properties are set

            if not finished:
                finished = time.time()
            self._metrics["wtime"] = round(finished - self._started, 3)

        try:
            if timeout:
                process_is_finished = lambda: not self.running
                fail_message = "Command '%s' stopped by %d seconds timeout" % (self._command, timeout)
                try:
                    wait_for(process_is_finished, timeout, fail_message, sleep_time=0.1, on_check_condition=self._process_progress_listener)
                except TimeoutError as e:
                    if on_timeout:
                        yatest_logger.debug("Calling user specified on_timeout function")
                        try:
                            on_timeout(self, timeout)
                        except Exception:
                            yatest_logger.exception("Exception while calling on_timeout")
                    raise ExecutionTimeoutError(self, str(e))
            # Wait should be always called here, it finalizes internal states of its process and sets up return code
            _wait()
        except BaseException as e:
            _kill_process_tree(self._process.pid)
            _wait()
            yatest_logger.debug("Command exception: %s", e)
            raise
        finally:
            self._elapsed = time.time() - self._start
            self._save_outputs()
            self.verify_no_coredumps()

        self._finalise(check_exit_code)

    def _finalise(self, check_exit_code):
        # Set the signal (negative number) which caused the process to exit
        if check_exit_code and self.exit_code != 0:
            yatest_logger.error("Execution failed with exit code: %s\n\t,std_out:%s\n\tstd_err:%s\n",
                                self.exit_code, truncate(self.std_out, MAX_OUT_LEN), truncate(self.std_err, MAX_OUT_LEN))
            raise ExecutionError(self)

        # Don't search for sanitize errors if stderr was redirected
        self.verify_sanitize_errors()

    def verify_no_coredumps(self):
        """
        Verify there is no coredump from this binary. If there is then report backtrace.
        """
        if self.exit_code < 0 and self._collect_cores:
            if cores:
                try:
                    self._recover_core()
                except Exception:
                    yatest_logger.exception("Exception while recovering core")
            else:
                yatest_logger.warning("Core dump file recovering is skipped: module cores isn't available")

    def verify_sanitize_errors(self):
        """
        Verify there are no sanitizer (ASAN, MSAN, TSAN, etc) errors for this binary. If there are any report them.
        """
        if self._std_err and self._check_sanitizer and runtime._get_ya_config().sanitizer_extra_checks:
            build_path = runtime.build_path()
            if self.command[0].startswith(build_path):
                match = re.search(SANITIZER_ERROR_PATTERN, self._std_err)
                if match:
                    yatest_logger.error("%s sanitizer found errors:\n\tstd_err:%s\n", match.group(1), truncate(self.std_err, MAX_OUT_LEN))
                    raise ExecutionError(self)
                else:
                    yatest_logger.debug("No sanitizer errors found")
            else:
                yatest_logger.debug("'%s' doesn't belong to '%s' - no check for sanitize errors", self.command[0], build_path)


# Don't forget to sync changes in the interface and defaults with yatest.yt.process.execute
def execute(
    command, check_exit_code=True,
    shell=False, timeout=None,
    cwd=None, env=None,
    stdin=None, stdout=None, stderr=None,
    creationflags=0, wait=True,
    process_progress_listener=None, close_fds=False,
    collect_cores=True, check_sanitizer=True, preexec_fn=None, on_timeout=None,
    executor=_Execution,
):
    """
    Executes a command
    :param command: command: can be a list of arguments or a string
    :param check_exit_code: will raise ExecutionError if the command exits with non zero code
    :param shell: use shell to run the command
    :param timeout: execution timeout
    :param cwd: working directory
    :param env: command environment
    :param stdin: command stdin
    :param stdout: command stdout
    :param stderr: command stderr
    :param creationflags: command creation flags
    :param wait: should wait until the command finishes
    :param process_progress_listener=object that is polled while execution is in progress
    :param close_fds:  subrpocess.Popen close_fds args
    :param collect_cores: recover core dump files if shell == False
    :param check_sanitizer: raise ExecutionError if stderr contains sanitize errors
    :param preexec_fn: subrpocess.Popen preexec_fn arg
    :param on_timeout: on_timeout(<execution object>, <timeout value>) callback

    :return: Execution object
    """
    if env is None:
        env = os.environ.copy()
    else:
        mandatory_system_vars = ["TMPDIR"]
        for var in mandatory_system_vars:
            if var not in env and var in os.environ:
                env[var] = os.environ[var]

    if not wait and timeout is not None:
        raise ValueError("Incompatible arguments 'timeout' and wait=False")

    # if subprocess.PIPE in [stdout, stderr]:
    #     raise ValueError("Don't use pipe to obtain stream data - it may leads to the deadlock")

    def get_out_stream(stream, default_name):
        if stream is None:
            # No stream is supplied: open new temp file
            return _get_command_output_file(command, default_name), False

        if isinstance(stream, six.string_types):
            # User filename is supplied: open file for writing
            return open(stream, 'wb+'), stream.startswith('/dev/')

        # Open file or PIPE sentinel is supplied
        is_pipe = stream == subprocess.PIPE
        return stream, not is_pipe

    # to be able to have stdout/stderr and track the process time execution, we don't use subprocess.PIPE,
    # as it can cause processes hangs, but use tempfiles instead
    out_file, user_stdout = get_out_stream(stdout, 'out')
    err_file, user_stderr = get_out_stream(stderr, 'err')
    in_file = stdin

    if shell and type(command) == list:
        command = " ".join(command)

    if shell:
        collect_cores = False
        check_sanitizer = False

    if check_sanitizer:
        env["LSAN_OPTIONS"] = environment.extend_env_var(os.environ, "LSAN_OPTIONS", "exitcode=100")

    if stdin:
        name = "PIPE" if stdin == subprocess.PIPE else stdin.name
        yatest_logger.debug("Executing '%s' with input '%s' in '%s'", command, name, cwd)
    else:
        yatest_logger.debug("Executing '%s' in '%s'", command, cwd)
    # XXX

    started = time.time()
    process = subprocess.Popen(
        command, shell=shell, universal_newlines=True,
        stdout=out_file, stderr=err_file, stdin=in_file,
        cwd=cwd, env=env, creationflags=creationflags, close_fds=close_fds, preexec_fn=preexec_fn,
    )
    yatest_logger.debug("Command pid: %s", process.pid)

    res = executor(command, process, out_file, err_file, process_progress_listener, cwd, collect_cores, check_sanitizer, started, user_stdout=user_stdout, user_stderr=user_stderr)
    if wait:
        res.wait(check_exit_code, timeout, on_timeout)
    return res


def _get_command_output_file(cmd, ext):
    command_name = get_command_name(cmd)
    file_name = command_name + "." + ext
    try:
        # if execution is performed from test, save out / err to the test logs dir
        import yatest.common
        import pytest
        if not hasattr(pytest, 'config'):
            raise ImportError("not in test")
        file_name = path.get_unique_file_path(yatest.common.output_path(), file_name)
        yatest_logger.debug("Command %s will be placed to %s", ext, os.path.basename(file_name))
        return open(file_name, "wb+")
    except ImportError:
        return tempfile.NamedTemporaryFile(delete=False, suffix=file_name)


def _get_proc_tree_info(pids):
    if os.name == 'nt':
        return 'Not supported'
    else:
        stdout, _ = subprocess.Popen(["/bin/ps", "-wufp"] + [str(p) for p in pids], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        return stdout


def py_execute(
    command, check_exit_code=True,
    shell=False, timeout=None,
    cwd=None, env=None,
    stdin=None, stdout=None, stderr=None,
    creationflags=0, wait=True,
    process_progress_listener=None, close_fds=False
):
    """
    Executes a command with the arcadia python
    :param command: command to pass to python
    :param check_exit_code: will raise ExecutionError if the command exits with non zero code
    :param shell: use shell to run the command
    :param timeout: execution timeout
    :param cwd: working directory
    :param env: command environment
    :param stdin: command stdin
    :param stdout: command stdout
    :param stderr: command stderr
    :param creationflags: command creation flags
    :param wait: should wait until the command finishes
    :param process_progress_listener=object that is polled while execution is in progress
    :return: Execution object
    """
    if isinstance(command, six.string_types):
        command = [command]
    command = [runtime.python_path()] + command
    if shell:
        command = " ".join(command)
    return execute(**locals())


def _format_error(error):
    return truncate(error, MAX_MESSAGE_LEN)


def wait_for(check_function, timeout, fail_message="", sleep_time=1.0, on_check_condition=None):
    """
    Tries to execute `check_function` for `timeout` seconds.
    Continue until function returns nonfalse value.
    If function doesn't return nonfalse value for `timeout` seconds
    OperationTimeoutException is raised.
    Return first nonfalse result returned by `checkFunction`.
    """
    if sleep_time <= 0:
        raise ValueError("Incorrect sleep time value {}".format(sleep_time))
    if timeout < 0:
        raise ValueError("Incorrect timeout value {}".format(timeout))
    start = time.time()
    while start + timeout > time.time():
        if on_check_condition:
            on_check_condition()

        res = check_function()
        if res:
            return res
        time.sleep(sleep_time)

    message = "{} second(s) wait timeout has expired".format(timeout)
    if fail_message:
        message += ": {}".format(fail_message)
    raise TimeoutError(truncate(message, MAX_MESSAGE_LEN))


def _kill_process_tree(process_pid, target_pid_signal=None):
    """
    Kills child processes, req. Note that psutil should be installed
    @param process_pid: parent id to search for descendants
    """
    yatest_logger.debug("Killing process %s", process_pid)
    if os.name == 'nt':
        _win_kill_process_tree(process_pid)
    else:
        _nix_kill_process_tree(process_pid, target_pid_signal)


def _nix_get_proc_children(pid):
    try:
        cmd = ["pgrep", "-P", str(pid)]
        return [int(p) for p in subprocess.check_output(cmd).split()]
    except Exception:
        return []


def _get_binname(pid):
    try:
        return os.path.basename(os.readlink('/proc/{}/exe'.format(pid)))
    except Exception as e:
        return "error({})".format(e)


def _nix_kill_process_tree(pid, target_pid_signal=None):
    """
    Kills the process tree.
    """
    yatest_logger.debug("Killing process tree for pid {} (bin:'{}')".format(pid, _get_binname(pid)))

    def try_to_send_signal(pid, sig):
        try:
            os.kill(pid, sig)
            yatest_logger.debug("Sent signal %d to the pid %d", sig, pid)
        except Exception as exc:
            yatest_logger.debug("Error while sending signal {sig} to pid {pid}: {error}".format(sig=sig, pid=pid, error=str(exc)))

    try_to_send_signal(pid, signal.SIGSTOP)  # Stop the process to prevent it from starting any child processes.

    # Get the child process PID list.
    child_pids = _nix_get_proc_children(pid)
    # Stop the child processes.
    for child_pid in child_pids:
        try:
            # Kill the child recursively.
            _kill_process_tree(int(child_pid))
        except Exception as e:
            # Skip the error and continue killing.
            yatest_logger.debug("Killing child pid {pid} failed: {error}".format(pid=child_pid, error=e))
            continue

    try_to_send_signal(pid, target_pid_signal or signal.SIGKILL)  # Kill the root process.

    # sometimes on freebsd sigkill cannot kill the process and either sigkill or sigcont should be sent
    # https://www.mail-archive.com/freebsd-hackers@freebsd.org/msg159646.html
    try_to_send_signal(pid, signal.SIGCONT)


def _win_kill_process_tree(pid):
    subprocess.call(['taskkill', '/F', '/T', '/PID', str(pid)])


def _run_readelf(binary_path):
    return str(subprocess.check_output([runtime.binary_path('contrib/python/pyelftools/readelf/readelf'), '-s', runtime.binary_path(binary_path)]))


def check_glibc_version(binary_path):
    lucid_glibc_version = distutils.version.LooseVersion("2.11")

    for l in _run_readelf(binary_path).split('\n'):
        match = GLIBC_PATTERN.search(l)
        if not match:
            continue
        assert distutils.version.LooseVersion(match.group(1)) <= lucid_glibc_version, match.group(0)


def backtrace_to_html(bt_filename, output):
    try:
        from library.python.coredump_filter import core_proc
        with open(output, "wb") as afile:
            core_proc.filter_stackdump(bt_filename, stream=afile)
    except ImportError as e:
        yatest_logger.debug("Failed to import coredump_filter: %s", e)

        with open(output, "wb") as afile:
            res = execute([runtime.python_path(), runtime.source_path("library/python/coredump_filter/core_proc.py"), bt_filename], check_exit_code=False, check_sanitizer=False, stdout=afile)
        if res.exit_code != 0:
            with open(output, "ab") as afile:
                afile.write("\n")
                afile.write(res.std_err)
