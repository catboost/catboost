# coding: utf-8

import os
import re
import sys
import time
import types
import signal
import shutil
import logging
import tempfile
import subprocess
import errno

import cores
import runtime
import path
import environment


MAX_OUT_LEN = 1000 * 1000  # 1 mb
MAX_MESSAGE_LEN = 1500
SANITIZER_ERROR_PATTERN = r": ([A-Z][\w]+Sanitizer)"
yatest_logger = logging.getLogger("ya.test")


def truncate(s, size):
    if s is None:
        return None
    elif len(s) <= size:
        return s
    else:
        return "..." + s[-(size - 3):]


def get_command_name(command):
    return os.path.basename(command.split()[0] if isinstance(command, types.StringTypes) else command[0])


class ExecutionError(Exception):

    def __init__(self, execution_result):
        if not isinstance(execution_result.command, basestring):
            command = " ".join(str(arg) for arg in execution_result.command)
        else:
            command = execution_result.command
        message = "Command '{command}' has failed with code {code}.\nErrors:\n{err}\n".format(
            command=command,
            code=execution_result.exit_code,
            err=_format_error(execution_result.std_err))
        if execution_result.backtrace:
            message += "Backtrace:\n[[rst]]{}[[bad]]\n".format(cores.colorize_backtrace(execution_result._backtrace))
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


class _Execution(object):

    def __init__(self, command, process, out_file, err_file, process_progress_listener=None, cwd=None, collect_cores=True, check_sanitizer=True, started=0):
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

    @property
    def process(self):
        return self._process

    @property
    def command(self):
        return self._command

    @property
    def exit_code(self):
        return self._process.returncode

    @property
    def std_out(self):
        if self._process.stdout and not self._std_out:
            self._std_out = self._process.stdout.read()
        if self._std_out is not None:
            return self._std_out

    @property
    def std_err(self):
        if self._process.stderr and not self._std_err:
            self._std_err = self._process.stderr.read()
        if self._std_err is not None:
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
        if self._out_file:
            self._out_file.flush()
            self._out_file.seek(0, os.SEEK_SET)
            self._std_out = self._out_file.read()
        if self._err_file:
            self._err_file.flush()
            self._err_file.seek(0, os.SEEK_SET)
            self._std_err = self._err_file.read()
        if clean_files:
            self._clean_files()
        yatest_logger.debug("Command (pid %s) rc: %s", self._process.pid, self.exit_code)
        yatest_logger.debug("Command (pid %s) elapsed time (sec): %s", self._process.pid, self.elapsed)
        if self._metrics:
            for key, value in self._metrics.iteritems():
                yatest_logger.debug("Command (pid %s) %s: %s", self._process.pid, key, value)
        yatest_logger.debug("Command (pid %s) output:\n%s", self._process.pid, truncate(self._std_out, MAX_OUT_LEN))
        yatest_logger.debug("Command (pid %s) errors:\n%s", self._process.pid, truncate(self._std_err, MAX_OUT_LEN))

    def _clean_files(self):
        open_files = []
        if self._err_file:
            open_files.append(self._err_file)
        if self._out_file:
            open_files.append(self._out_file)
        for f in open_files:
            f.close()

    def _recover_core(self):
        core_path = cores.recover_core_dump_file(self.command[0], self._cwd, self.process.pid)
        if core_path:
            # Core dump file recovering may be disabled (for distbuild for example) - produce only bt
            store_cores = runtime._get_ya_config().collect_cores
            if store_cores:
                new_core_path = path.get_unique_file_path(runtime.output_path(), "{}.core".format(os.path.basename(self.command[0])))
                # Copy core dump file, because it may be overwritten
                yatest_logger.debug("Coping core dump file from '%s' to the '%s'", core_path, new_core_path)
                shutil.copyfile(core_path, new_core_path)
                core_path = new_core_path

            bt_filename = None
            pbt_filename = None

            if os.path.exists(runtime.gdb_path()):
                self._backtrace = cores.get_gdb_full_backtrace(self.command[0], core_path, runtime.gdb_path())
                bt_filename = path.get_unique_file_path(runtime.output_path(), "{}.backtrace".format(os.path.basename(self.command[0])))
                with open(bt_filename, "w") as afile:
                    afile.write(self._backtrace)
                # generate pretty html version of backtrace aka Tri Korochki
                pbt_filename = bt_filename + ".html"
                cores.backtrace_to_html(bt_filename, pbt_filename)

            if store_cores:
                runtime._register_core(os.path.basename(self.command[0]), self.command[0], core_path, bt_filename, pbt_filename)
            else:
                runtime._register_core(os.path.basename(self.command[0]), None, None, bt_filename, pbt_filename)

    def wait(self, check_exit_code=True, timeout=None, on_timeout=None):

        def _wait():
            finished = None
            try:
                if hasattr(os, "wait4"):
                    try:
                        pid, sts, rusage = subprocess._eintr_retry_call(os.wait4, self._process.pid, 0)
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
            finally:
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

            if self.exit_code < 0 and self._collect_cores:
                try:
                    self._recover_core()
                except Exception:
                    yatest_logger.exception("Exception while recovering core")

        # Set the signal (negative number) which caused the process to exit
        if check_exit_code and self.exit_code != 0:
            yatest_logger.error("Execution failed with exit code: %s\n\t,std_out:%s\n\tstd_err:%s\n",
                                self.exit_code, truncate(self.std_out, MAX_OUT_LEN), truncate(self.std_err, MAX_OUT_LEN))
            raise ExecutionError(self)

        # Don't search for sanitize errors if stderr was redirected
        if self._std_err and self._check_sanitizer and runtime._get_ya_config().sanitizer_extra_checks:
            build_path = runtime.build_path()
            if self.command[0].startswith(build_path):
                match = re.search(SANITIZER_ERROR_PATTERN, self._std_err)
                if match:
                    yatest_logger.error("%s sanitizer found errors:\n\tstd_err:%s\n", match.group(1), truncate(self.std_err, MAX_OUT_LEN))
                    raise ExecutionError(self)
                    yatest_logger.debug("No sanitizer errors found")
            else:
                yatest_logger.debug("'%s' doesn't belong to '%s' - no check for sanitize errors", self.command[0], build_path)


def execute(
    command, check_exit_code=True,
    shell=False, timeout=None,
    cwd=None, env=None,
    stdin=None, stdout=None, stderr=None,
    creationflags=0, wait=True,
    process_progress_listener=None, close_fds=False,
    collect_cores=True, check_sanitizer=True, preexec_fn=None, on_timeout=None,
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
    if not wait and timeout is not None:
        raise ValueError("Incompatible arguments 'timeout' and wait=False")

    def get_temp_file(ext):
        command_name = get_command_name(command)
        file_name = command_name + "." + ext
        try:
            # if execution is performed from test, save out / err to the test logs dir
            import yatest.common
            import pytest
            if not hasattr(pytest, 'config'):
                raise ImportError("not in test")
            file_name = path.get_unique_file_path(yatest.common.output_path(), file_name)
            yatest_logger.debug("Command %s will be placed to %s", ext, os.path.basename(file_name))
            return open(file_name, "w+")
        except ImportError:
            return tempfile.NamedTemporaryFile(delete=False, suffix=file_name)

    # to be able to have stdout/stderr and track the process time execution, we don't use subprocess.PIPE,
    # as it can cause processes hangs, but use tempfiles instead
    out_file = stdout or get_temp_file("out")
    err_file = stderr or get_temp_file("err")
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
    try:
        process = subprocess.Popen(command, shell=shell, universal_newlines=True,
                                   stdout=out_file, stderr=err_file, stdin=in_file,
                                   cwd=cwd, env=env, creationflags=creationflags, close_fds=close_fds, preexec_fn=preexec_fn)
        yatest_logger.debug("Command pid: %s", process.pid)
    except OSError as e:
        # XXX
        # Trying to catch 'Text file busy' issue
        if e.errno == 26:
            try:
                message = _get_oserror26_exception_message(command)
            except Exception as newe:
                yatest_logger.error(str(newe))
            else:
                raise type(e), type(e)(e.message + message), sys.exc_info()[2]
        raise e
    res = _Execution(command, process, not stdout and out_file, not stderr and err_file, process_progress_listener, cwd, collect_cores, check_sanitizer, started)
    if wait:
        res.wait(check_exit_code, timeout, on_timeout)
    return res


def _get_proc_tree_info(pids):
    if os.name == 'nt':
        return 'Not supported'
    else:
        stdout, _ = subprocess.Popen(["/bin/ps", "-wufp"] + [str(p) for p in pids], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        return stdout


# XXX
def _get_oserror26_exception_message(command):

    def get_ppid(pid):
        stdout, _ = subprocess.Popen(["/bin/ps", "-o", "ppid=", "-p", str(pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        return int(stdout.strip())

    stdout, _ = subprocess.Popen(["/usr/bin/lsof", command[0]], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    yatest_logger.debug("lsof %s: %s", command[0], stdout)

    stdout, stderr = subprocess.Popen(["/usr/bin/lsof", '-Fp', command[0]], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    if stderr:
        raise Exception(stderr)
    message = "[Errno 26] Text file busy\n\nProcesses holding {}:\n".format(command[0])
    for line in stdout.strip().split("\n"):
        # lsof format is pPID
        pid = int(line[1:])
        pids = [pid]
        while pid != 1:
            pid = get_ppid(pid)
            pids.append(pid)

        message += _get_proc_tree_info(pids)
    return message + "\nPlease, report to the devtools@yandex-team.ru about this issue"


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
    if isinstance(command, types.StringTypes):
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


def _nix_kill_process_tree(pid, target_pid_signal=None):
    """
    Kills the process tree.
    """
    yatest_logger.debug("Killing process tree for pid {pid}".format(pid=pid))

    def try_to_send_signal(pid, sig):
        try:
            os.kill(pid, sig)
            yatest_logger.debug("Sent signal %d to the pid %d", sig, pid)
        except Exception as exc:
            yatest_logger.debug("Error while sending signal {sig} to pid {pid}: {error}".format(sig=sig, pid=pid, error=str(exc)))

    try_to_send_signal(pid, signal.SIGSTOP)  # Stop the process to prevent it from starting any child processes.

    # Get the child process PID list.
    try:
        pgrep_command = ["pgrep", "-P", str(pid)]
        child_pids = subprocess.check_output(pgrep_command).split()
    except Exception:
        yatest_logger.debug("Process with pid {pid} does not have child processes".format(pid=pid))
    else:
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
