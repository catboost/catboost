# subprocess - Subprocesses with accessible I/O streams
#
# For more information about this module, see PEP 324.
#
# Copyright (c) 2003-2005 by Peter Astrand <astrand@lysator.liu.se>
#
# Licensed to PSF under a Contributor Agreement.
# See http://www.python.org/3.3/license for licensing details.

r"""Subprocesses with accessible I/O streams

This module allows you to spawn processes, connect to their
input/output/error pipes, and obtain their return codes.

For a complete description of this module see the Python documentation.

Main API
========
run(...): Runs a command, waits for it to complete, then returns a
          CompletedProcess instance.
Popen(...): A class for flexibly executing a command in a new process

Constants
---------
DEVNULL: Special value that indicates that os.devnull should be used
PIPE:    Special value that indicates a pipe should be created
STDOUT:  Special value that indicates that stderr should go to stdout


Older API
=========
call(...): Runs a command, waits for it to complete, then returns
    the return code.
check_call(...): Same as call() but raises CalledProcessError()
    if return code is not 0
check_output(...): Same as check_call() but returns the contents of
    stdout instead of a return code
"""

import sys
mswindows = (sys.platform == "win32")

import os
import errno
import exceptions
import types
import time
import traceback
import gc
import signal

# Exception classes used by this module.
class SubprocessError(Exception): pass


class CalledProcessError(SubprocessError):
    """Raised when run() is called with check=True and the process
    returns a non-zero exit status.

    Attributes:
      cmd, returncode, stdout, stderr, output
    """
    def __init__(self, returncode, cmd, output=None, stderr=None):
        self.returncode = returncode
        self.cmd = cmd
        self.output = output
        self.stderr = stderr
        super(CalledProcessError, self).__init__(returncode, cmd,
                                                 output, stderr)

    def __str__(self):
        if self.returncode and self.returncode < 0:
            return "Command '%s' died with signal %d." % (
                    self.cmd, -self.returncode)
        else:
            return "Command '%s' returned non-zero exit status %d." % (
                    self.cmd, self.returncode)

    #@property
    def __stdout_getter(self):
        """Alias for output attribute, to match stderr"""
        return self.output

    #@stdout.setter  # Required Python 2.6
    def __stdout_setter(self, value):
        # There's no obvious reason to set this, but allow it anyway so
        # .stdout is a transparent alias for .output
        self.output = value

    stdout = property(__stdout_getter, __stdout_setter)  # Python 2.4


class TimeoutExpired(SubprocessError):
    """This exception is raised when the timeout expires while waiting for a
    child process.

    Attributes:
        cmd, output, stdout, stderr, timeout
    """
    def __init__(self, cmd, timeout, output=None, stderr=None):
        self.cmd = cmd
        self.timeout = timeout
        self.output = output
        self.stderr = stderr
        super(TimeoutExpired, self).__init__(cmd, timeout, output, stderr)

    def __str__(self):
        return ("Command '%s' timed out after %s seconds" %
                (self.cmd, self.timeout))

    #@property
    def __stdout_getter(self):
        return self.output

    #@stdout.setter  # Required Python 2.6
    def __stdout_setter(self, value):
        # There's no obvious reason to set this, but allow it anyway so
        # .stdout is a transparent alias for .output
        self.output = value

    stdout = property(__stdout_getter, __stdout_setter)  # Python 2.4


if mswindows:
    import threading
    import msvcrt
    import _subprocess
    class STARTUPINFO:
        dwFlags = 0
        hStdInput = None
        hStdOutput = None
        hStdError = None
        wShowWindow = 0
    class pywintypes:
        error = IOError
else:
    import select
    _has_poll = hasattr(select, 'poll')
    import fcntl
    import pickle

    try:
        import _posixsubprocess32 as _posixsubprocess
    except ImportError:
        _posixsubprocess = None
        import warnings
        warnings.warn("The _posixsubprocess module is not being used. "
                      "Child process reliability may suffer if your "
                      "program uses threads.", RuntimeWarning)
    try:
        import threading
    except ImportError:
        import dummy_threading as threading

    # When select or poll has indicated that the file is writable,
    # we can write up to _PIPE_BUF bytes without risk of blocking.
    # POSIX defines PIPE_BUF as >= 512.
    _PIPE_BUF = getattr(select, 'PIPE_BUF', 512)

    _FD_CLOEXEC = getattr(fcntl, 'FD_CLOEXEC', 1)

    def _set_cloexec(fd, cloexec):
        old = fcntl.fcntl(fd, fcntl.F_GETFD)
        if cloexec:
            fcntl.fcntl(fd, fcntl.F_SETFD, old | _FD_CLOEXEC)
        else:
            fcntl.fcntl(fd, fcntl.F_SETFD, old & ~_FD_CLOEXEC)

    if _posixsubprocess:
        _create_pipe = _posixsubprocess.cloexec_pipe
    else:
        def _create_pipe():
            fds = os.pipe()
            _set_cloexec(fds[0], True)
            _set_cloexec(fds[1], True)
            return fds

__all__ = ["Popen", "PIPE", "STDOUT", "call", "check_call",
           "check_output", "CalledProcessError"]

if mswindows:
    from _subprocess import (CREATE_NEW_CONSOLE, CREATE_NEW_PROCESS_GROUP,
                             STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
                             STD_ERROR_HANDLE, SW_HIDE,
                             STARTF_USESTDHANDLES, STARTF_USESHOWWINDOW)
    # https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032(v=vs.85).aspx
    # Note: In Python 3.3 this constant is found in the _winapi module.
    _WAIT_TIMEOUT = 0x102

    __all__.extend(["CREATE_NEW_CONSOLE", "CREATE_NEW_PROCESS_GROUP",
                    "STD_INPUT_HANDLE", "STD_OUTPUT_HANDLE",
                    "STD_ERROR_HANDLE", "SW_HIDE",
                    "STARTF_USESTDHANDLES", "STARTF_USESHOWWINDOW"])
try:
    MAXFD = os.sysconf("SC_OPEN_MAX")
except:
    MAXFD = 256

# This lists holds Popen instances for which the underlying process had not
# exited at the time its __del__ method got called: those processes are wait()ed
# for synchronously from _cleanup() when a new Popen object is created, to avoid
# zombie processes.
_active = []

def _cleanup():
    for inst in _active[:]:
        res = inst._internal_poll(_deadstate=sys.maxint)
        if res is not None:
            try:
                _active.remove(inst)
            except ValueError:
                # This can happen if two threads create a new Popen instance.
                # It's harmless that it was already removed, so ignore.
                pass

PIPE = -1
STDOUT = -2
DEVNULL = -3

# This function is only used by multiprocessing, it is here so that people
# can drop subprocess32 in as a replacement for the stdlib subprocess module.

def _args_from_interpreter_flags():
    """Return a list of command-line arguments reproducing the current
    settings in sys.flags and sys.warnoptions."""
    flag_opt_map = {
        'debug': 'd',
        # 'inspect': 'i',
        # 'interactive': 'i',
        'optimize': 'O',
        'dont_write_bytecode': 'B',
        'no_user_site': 's',
        'no_site': 'S',
        'ignore_environment': 'E',
        'verbose': 'v',
        'bytes_warning': 'b',
        'py3k_warning': '3',
    }
    args = []
    for flag, opt in flag_opt_map.items():
        v = getattr(sys.flags, flag)
        if v > 0:
            args.append('-' + opt * v)
    if getattr(sys.flags, 'hash_randomization') != 0:
        args.append('-R')
    for opt in sys.warnoptions:
        args.append('-W' + opt)
    return args


def _eintr_retry_call(func, *args):
    while True:
        try:
            return func(*args)
        except (OSError, IOError), e:
            if e.errno == errno.EINTR:
                continue
            raise


def _get_exec_path(env=None):
    """Returns the sequence of directories that will be searched for the
    named executable (similar to a shell) when launching a process.

    *env* must be an environment variable dict or None.  If *env* is None,
    os.environ will be used.
    """
    if env is None:
        env = os.environ
    return env.get('PATH', os.defpath).split(os.pathsep)


if hasattr(os, 'get_exec_path'):
    _get_exec_path = os.get_exec_path


def call(*popenargs, **kwargs):
    """Run command with arguments.  Wait for command to complete or
    timeout, then return the returncode attribute.

    The arguments are the same as for the Popen constructor.  Example:

    retcode = call(["ls", "-l"])
    """
    timeout = kwargs.pop('timeout', None)
    p = Popen(*popenargs, **kwargs)
    try:
        return p.wait(timeout=timeout)
    except TimeoutExpired:
        p.kill()
        p.wait()
        raise


def check_call(*popenargs, **kwargs):
    """Run command with arguments.  Wait for command to complete.  If
    the exit code was zero then return, otherwise raise
    CalledProcessError.  The CalledProcessError object will have the
    return code in the returncode attribute.

    The arguments are the same as for the call function.  Example:

    check_call(["ls", "-l"])
    """
    retcode = call(*popenargs, **kwargs)
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise CalledProcessError(retcode, cmd)
    return 0


def check_output(*popenargs, **kwargs):
    r"""Run command with arguments and return its output as a byte string.

    If the exit code was non-zero it raises a CalledProcessError.  The
    CalledProcessError object will have the return code in the returncode
    attribute and output in the output attribute.

    The arguments are the same as for the Popen constructor.  Example:

    >>> check_output(["ls", "-l", "/dev/null"])
    'crw-rw-rw- 1 root root 1, 3 Oct 18  2007 /dev/null\n'

    The stdout argument is not allowed as it is used internally.
    To capture standard error in the result, use stderr=STDOUT.

    >>> check_output(["/bin/sh", "-c",
    ...               "ls -l non_existent_file ; exit 0"],
    ...              stderr=STDOUT)
    'ls: non_existent_file: No such file or directory\n'
    """
    timeout = kwargs.pop('timeout', None)
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = Popen(stdout=PIPE, *popenargs, **kwargs)
    try:
        output, unused_err = process.communicate(timeout=timeout)
    except TimeoutExpired:
        process.kill()
        output, unused_err = process.communicate()
        raise TimeoutExpired(process.args, timeout, output=output)
    retcode = process.poll()
    if retcode:
        raise CalledProcessError(retcode, process.args, output=output)
    return output


class CompletedProcess(object):
    """A process that has finished running.
    This is returned by run().
    Attributes:
      args: The list or str args passed to run().
      returncode: The exit code of the process, negative for signals.
      stdout: The standard output (None if not captured).
      stderr: The standard error (None if not captured).
    """
    def __init__(self, args, returncode, stdout=None, stderr=None):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __repr__(self):
        args = ['args={!r}'.format(self.args),
                'returncode={!r}'.format(self.returncode)]
        if self.stdout is not None:
            args.append('stdout={!r}'.format(self.stdout))
        if self.stderr is not None:
            args.append('stderr={!r}'.format(self.stderr))
        return "{}({})".format(type(self).__name__, ', '.join(args))

    def check_returncode(self):
        """Raise CalledProcessError if the exit code is non-zero."""
        if self.returncode:
            raise CalledProcessError(self.returncode, self.args, self.stdout,
                                     self.stderr)


def run(*popenargs, **kwargs):
    """Run command with arguments and return a CompletedProcess instance.
    The returned instance will have attributes args, returncode, stdout and
    stderr. By default, stdout and stderr are not captured, and those attributes
    will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them.
    If check is True and the exit code was non-zero, it raises a
    CalledProcessError. The CalledProcessError object will have the return code
    in the returncode attribute, and output & stderr attributes if those streams
    were captured.
    If timeout is given, and the process takes too long, a TimeoutExpired
    exception will be raised.
    There is an optional argument "input", allowing you to
    pass a string to the subprocess's stdin.  If you use this argument
    you may not also use the Popen constructor's "stdin" argument, as
    it will be used internally.
    The other arguments are the same as for the Popen constructor.
    If universal_newlines=True is passed, the "input" argument must be a
    string and stdout/stderr in the returned object will be strings rather than
    bytes.
    """
    input = kwargs.pop('input', None)
    timeout = kwargs.pop('timeout', None)
    check = kwargs.pop('check', False)
    if input is not None:
        if 'stdin' in kwargs:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = PIPE

    process = Popen(*popenargs, **kwargs)
    try:
        process.__enter__()  # No-Op really... illustrate "with in 2.4"
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise TimeoutExpired(process.args, timeout, output=stdout,
                                 stderr=stderr)
        except:
            process.kill()
            process.wait()
            raise
        retcode = process.poll()
        if check and retcode:
            raise CalledProcessError(retcode, process.args,
                                     output=stdout, stderr=stderr)
    finally:
        # None because our context manager __exit__ does not use them.
        process.__exit__(None, None, None)
    return CompletedProcess(process.args, retcode, stdout, stderr)


def list2cmdline(seq):
    """
    Translate a sequence of arguments into a command line
    string, using the same rules as the MS C runtime:

    1) Arguments are delimited by white space, which is either a
       space or a tab.

    2) A string surrounded by double quotation marks is
       interpreted as a single argument, regardless of white space
       contained within.  A quoted string can be embedded in an
       argument.

    3) A double quotation mark preceded by a backslash is
       interpreted as a literal double quotation mark.

    4) Backslashes are interpreted literally, unless they
       immediately precede a double quotation mark.

    5) If backslashes immediately precede a double quotation mark,
       every pair of backslashes is interpreted as a literal
       backslash.  If the number of backslashes is odd, the last
       backslash escapes the next double quotation mark as
       described in rule 3.
    """

    # See
    # http://msdn.microsoft.com/en-us/library/17w5ykft.aspx
    # or search http://msdn.microsoft.com for
    # "Parsing C++ Command-Line Arguments"
    result = []
    needquote = False
    for arg in seq:
        bs_buf = []

        # Add a space to separate this argument from the others
        if result:
            result.append(' ')

        needquote = (" " in arg) or ("\t" in arg) or not arg
        if needquote:
            result.append('"')

        for c in arg:
            if c == '\\':
                # Don't know if we need to double yet.
                bs_buf.append(c)
            elif c == '"':
                # Double backslashes.
                result.append('\\' * len(bs_buf)*2)
                bs_buf = []
                result.append('\\"')
            else:
                # Normal char
                if bs_buf:
                    result.extend(bs_buf)
                    bs_buf = []
                result.append(c)

        # Add remaining backslashes, if any.
        if bs_buf:
            result.extend(bs_buf)

        if needquote:
            result.extend(bs_buf)
            result.append('"')

    return ''.join(result)


_PLATFORM_DEFAULT_CLOSE_FDS = object()


class Popen(object):
    def __init__(self, args, bufsize=0, executable=None,
                 stdin=None, stdout=None, stderr=None,
                 preexec_fn=None, close_fds=_PLATFORM_DEFAULT_CLOSE_FDS,
                 shell=False, cwd=None, env=None, universal_newlines=False,
                 startupinfo=None, creationflags=0,
                 restore_signals=True, start_new_session=False,
                 pass_fds=()):
        """Create new Popen instance."""
        _cleanup()
        # Held while anything is calling waitpid before returncode has been
        # updated to prevent clobbering returncode if wait() or poll() are
        # called from multiple threads at once.  After acquiring the lock,
        # code must re-check self.returncode to see if another thread just
        # finished a waitpid() call.
        self._waitpid_lock = threading.Lock()

        self._child_created = False
        self._input = None
        self._communication_started = False
        if not isinstance(bufsize, (int, long)):
            raise TypeError("bufsize must be an integer")

        if mswindows:
            if preexec_fn is not None:
                raise ValueError("preexec_fn is not supported on Windows "
                                 "platforms")
            any_stdio_set = (stdin is not None or stdout is not None or
                             stderr is not None)
            if close_fds is _PLATFORM_DEFAULT_CLOSE_FDS:
                if any_stdio_set:
                    close_fds = False
                else:
                    close_fds = True
            elif close_fds and any_stdio_set:
                raise ValueError(
                        "close_fds is not supported on Windows platforms"
                        " if you redirect stdin/stdout/stderr")
        else:
            # POSIX
            if close_fds is _PLATFORM_DEFAULT_CLOSE_FDS:
                close_fds = True
            if pass_fds and not close_fds:
                import warnings
                warnings.warn("pass_fds overriding close_fds.", RuntimeWarning)
                close_fds = True
            if startupinfo is not None:
                raise ValueError("startupinfo is only supported on Windows "
                                 "platforms")
            if creationflags != 0:
                raise ValueError("creationflags is only supported on Windows "
                                 "platforms")

        self.args = args
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self.pid = None
        self.returncode = None
        self.universal_newlines = universal_newlines

        # Input and output objects. The general principle is like
        # this:
        #
        # Parent                   Child
        # ------                   -----
        # p2cwrite   ---stdin--->  p2cread
        # c2pread    <--stdout---  c2pwrite
        # errread    <--stderr---  errwrite
        #
        # On POSIX, the child objects are file descriptors.  On
        # Windows, these are Windows file handles.  The parent objects
        # are file descriptors on both platforms.  The parent objects
        # are -1 when not using PIPEs. The child objects are -1
        # when not redirecting.

        (p2cread, p2cwrite,
         c2pread, c2pwrite,
         errread, errwrite) = self._get_handles(stdin, stdout, stderr)

        if mswindows:
            if p2cwrite != -1:
                p2cwrite = msvcrt.open_osfhandle(p2cwrite.Detach(), 0)
            if c2pread != -1:
                c2pread = msvcrt.open_osfhandle(c2pread.Detach(), 0)
            if errread != -1:
                errread = msvcrt.open_osfhandle(errread.Detach(), 0)

        if p2cwrite != -1:
            self.stdin = os.fdopen(p2cwrite, 'wb', bufsize)
        if c2pread != -1:
            if universal_newlines:
                self.stdout = os.fdopen(c2pread, 'rU', bufsize)
            else:
                self.stdout = os.fdopen(c2pread, 'rb', bufsize)
        if errread != -1:
            if universal_newlines:
                self.stderr = os.fdopen(errread, 'rU', bufsize)
            else:
                self.stderr = os.fdopen(errread, 'rb', bufsize)

        self._closed_child_pipe_fds = False
        exception_cleanup_needed = False
        try:
            try:
                self._execute_child(args, executable, preexec_fn, close_fds,
                                    pass_fds, cwd, env, universal_newlines,
                                    startupinfo, creationflags, shell,
                                    p2cread, p2cwrite,
                                    c2pread, c2pwrite,
                                    errread, errwrite,
                                    restore_signals, start_new_session)
            except:
                # The cleanup is performed within the finally block rather
                # than simply within this except block before the raise so
                # that any exceptions raised and handled within it do not
                # clobber the exception context we want to propagate upwards.
                # This is only necessary in Python 2.
                exception_cleanup_needed = True
                raise
        finally:
            if exception_cleanup_needed:
                for f in filter(None, (self.stdin, self.stdout, self.stderr)):
                    try:
                        f.close()
                    except EnvironmentError:
                        pass  # Ignore EBADF or other errors

                if not self._closed_child_pipe_fds:
                    to_close = []
                    if stdin == PIPE:
                        to_close.append(p2cread)
                    if stdout == PIPE:
                        to_close.append(c2pwrite)
                    if stderr == PIPE:
                        to_close.append(errwrite)
                    if hasattr(self, '_devnull'):
                        to_close.append(self._devnull)
                    for fd in to_close:
                        try:
                            os.close(fd)
                        except EnvironmentError:
                            pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.stdout:
            self.stdout.close()
        if self.stderr:
            self.stderr.close()
        if self.stdin:
            self.stdin.close()
        # Wait for the process to terminate, to avoid zombies.
        self.wait()

    def _translate_newlines(self, data):
        data = data.replace("\r\n", "\n")
        data = data.replace("\r", "\n")
        return data


    def __del__(self, _maxint=sys.maxint, _active=_active):
        # If __init__ hasn't had a chance to execute (e.g. if it
        # was passed an undeclared keyword argument), we don't
        # have a _child_created attribute at all.
        if not getattr(self, '_child_created', False):
            # We didn't get to successfully create a child process.
            return
        # In case the child hasn't been waited on, check if it's done.
        self._internal_poll(_deadstate=_maxint)
        if self.returncode is None and _active is not None:
            # Child is still running, keep us alive until we can wait on it.
            _active.append(self)


    def _get_devnull(self):
        if not hasattr(self, '_devnull'):
            self._devnull = os.open(os.devnull, os.O_RDWR)
        return self._devnull

    def _stdin_write(self, input):
        if input:
            try:
                self.stdin.write(input)
            except EnvironmentError as e:
                if e.errno == errno.EPIPE:
                    # communicate() must ignore broken pipe error
                    pass
                elif e.errno == errno.EINVAL :
                    # bpo-19612, bpo-30418: On Windows, stdin.write() fails
                    # with EINVAL if the child process exited or if the child
                    # process is still running but closed the pipe.
                    pass
                else:
                    raise

        try:
            self.stdin.close()
        except EnvironmentError as e:
            if e.errno in (errno.EPIPE, errno.EINVAL):
                pass
            else:
                raise

    def communicate(self, input=None, timeout=None):
        """Interact with process: Send data to stdin.  Read data from
        stdout and stderr, until end-of-file is reached.  Wait for
        process to terminate.  The optional input argument should be a
        string to be sent to the child process, or None, if no data
        should be sent to the child.

        communicate() returns a tuple (stdout, stderr)."""

        if self._communication_started and input:
            raise ValueError("Cannot send input after starting communication")

        if timeout is not None:
            endtime = time.time() + timeout
        else:
            endtime = None

        # Optimization: If we are not worried about timeouts, we haven't
        # started communicating, and we have one or zero pipes, using select()
        # or threads is unnecessary.
        if (endtime is None and not self._communication_started and
            [self.stdin, self.stdout, self.stderr].count(None) >= 2):
            stdout = None
            stderr = None
            if self.stdin:
                self._stdin_write(input)
            elif self.stdout:
                stdout = _eintr_retry_call(self.stdout.read)
                self.stdout.close()
            elif self.stderr:
                stderr = _eintr_retry_call(self.stderr.read)
                self.stderr.close()
            self.wait()
            return (stdout, stderr)

        try:
            stdout, stderr = self._communicate(input, endtime, timeout)
        finally:
            self._communication_started = True

        sts = self.wait(timeout=self._remaining_time(endtime))

        return (stdout, stderr)


    def poll(self):
        return self._internal_poll()


    def _remaining_time(self, endtime):
        """Convenience for _communicate when computing timeouts."""
        if endtime is None:
            return None
        else:
            return endtime - time.time()


    def _check_timeout(self, endtime, orig_timeout):
        """Convenience for checking if a timeout has expired."""
        if endtime is None:
            return
        if time.time() > endtime:
            raise TimeoutExpired(self.args, orig_timeout)


    if mswindows:
        #
        # Windows methods
        #
        def _get_handles(self, stdin, stdout, stderr):
            """Construct and return tuple with IO objects:
            p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite
            """
            if stdin is None and stdout is None and stderr is None:
                return (-1, -1, -1, -1, -1, -1)

            p2cread, p2cwrite = -1, -1
            c2pread, c2pwrite = -1, -1
            errread, errwrite = -1, -1

            if stdin is None:
                p2cread = _subprocess.GetStdHandle(_subprocess.STD_INPUT_HANDLE)
                if p2cread is None:
                    p2cread, _ = _subprocess.CreatePipe(None, 0)
            elif stdin == PIPE:
                p2cread, p2cwrite = _subprocess.CreatePipe(None, 0)
            elif stdin == DEVNULL:
                p2cread = msvcrt.get_osfhandle(self._get_devnull())
            elif isinstance(stdin, int):
                p2cread = msvcrt.get_osfhandle(stdin)
            else:
                # Assuming file-like object
                p2cread = msvcrt.get_osfhandle(stdin.fileno())
            p2cread = self._make_inheritable(p2cread)

            if stdout is None:
                c2pwrite = _subprocess.GetStdHandle(_subprocess.STD_OUTPUT_HANDLE)
                if c2pwrite is None:
                    _, c2pwrite = _subprocess.CreatePipe(None, 0)
            elif stdout == PIPE:
                c2pread, c2pwrite = _subprocess.CreatePipe(None, 0)
            elif stdout == DEVNULL:
                c2pwrite = msvcrt.get_osfhandle(self._get_devnull())
            elif isinstance(stdout, int):
                c2pwrite = msvcrt.get_osfhandle(stdout)
            else:
                # Assuming file-like object
                c2pwrite = msvcrt.get_osfhandle(stdout.fileno())
            c2pwrite = self._make_inheritable(c2pwrite)

            if stderr is None:
                errwrite = _subprocess.GetStdHandle(_subprocess.STD_ERROR_HANDLE)
                if errwrite is None:
                    _, errwrite = _subprocess.CreatePipe(None, 0)
            elif stderr == PIPE:
                errread, errwrite = _subprocess.CreatePipe(None, 0)
            elif stderr == STDOUT:
                errwrite = c2pwrite
            elif stderr == DEVNULL:
                errwrite = msvcrt.get_osfhandle(self._get_devnull())
            elif isinstance(stderr, int):
                errwrite = msvcrt.get_osfhandle(stderr)
            else:
                # Assuming file-like object
                errwrite = msvcrt.get_osfhandle(stderr.fileno())
            errwrite = self._make_inheritable(errwrite)

            return (p2cread, p2cwrite,
                    c2pread, c2pwrite,
                    errread, errwrite)


        def _make_inheritable(self, handle):
            """Return a duplicate of handle, which is inheritable"""
            return _subprocess.DuplicateHandle(_subprocess.GetCurrentProcess(),
                                handle, _subprocess.GetCurrentProcess(), 0, 1,
                                _subprocess.DUPLICATE_SAME_ACCESS)


        def _find_w9xpopen(self):
            """Find and return absolut path to w9xpopen.exe"""
            w9xpopen = os.path.join(
                            os.path.dirname(_subprocess.GetModuleFileName(0)),
                                    "w9xpopen.exe")
            if not os.path.exists(w9xpopen):
                # Eeek - file-not-found - possibly an embedding
                # situation - see if we can locate it in sys.exec_prefix
                w9xpopen = os.path.join(os.path.dirname(sys.exec_prefix),
                                        "w9xpopen.exe")
                if not os.path.exists(w9xpopen):
                    raise RuntimeError("Cannot locate w9xpopen.exe, which is "
                                       "needed for Popen to work with your "
                                       "shell or platform.")
            return w9xpopen


        def _execute_child(self, args, executable, preexec_fn, close_fds,
                           pass_fds, cwd, env, universal_newlines,
                           startupinfo, creationflags, shell,
                           p2cread, p2cwrite,
                           c2pread, c2pwrite,
                           errread, errwrite,
                           unused_restore_signals, unused_start_new_session):
            """Execute program (MS Windows version)"""

            assert not pass_fds, "pass_fds not supported on Windows."

            if not isinstance(args, types.StringTypes):
                args = list2cmdline(args)

            # Process startup details
            if startupinfo is None:
                startupinfo = STARTUPINFO()
            if -1 not in (p2cread, c2pwrite, errwrite):
                startupinfo.dwFlags |= _subprocess.STARTF_USESTDHANDLES
                startupinfo.hStdInput = p2cread
                startupinfo.hStdOutput = c2pwrite
                startupinfo.hStdError = errwrite

            if shell:
                startupinfo.dwFlags |= _subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = _subprocess.SW_HIDE
                comspec = os.environ.get("COMSPEC", "cmd.exe")
                args = comspec + " /c " + '"%s"' % args
                if (_subprocess.GetVersion() >= 0x80000000L or
                        os.path.basename(comspec).lower() == "command.com"):
                    # Win9x, or using command.com on NT. We need to
                    # use the w9xpopen intermediate program. For more
                    # information, see KB Q150956
                    # (http://web.archive.org/web/20011105084002/http://support.microsoft.com/support/kb/articles/Q150/9/56.asp)
                    w9xpopen = self._find_w9xpopen()
                    args = '"%s" %s' % (w9xpopen, args)
                    # Not passing CREATE_NEW_CONSOLE has been known to
                    # cause random failures on win9x.  Specifically a
                    # dialog: "Your program accessed mem currently in
                    # use at xxx" and a hopeful warning about the
                    # stability of your system.  Cost is Ctrl+C wont
                    # kill children.
                    creationflags |= _subprocess.CREATE_NEW_CONSOLE

            # Start the process
            try:
                try:
                    hp, ht, pid, tid = _subprocess.CreateProcess(executable, args,
                                             # no special security
                                             None, None,
                                             int(not close_fds),
                                             creationflags,
                                             env,
                                             cwd,
                                             startupinfo)
                except pywintypes.error, e:
                    # Translate pywintypes.error to WindowsError, which is
                    # a subclass of OSError.  FIXME: We should really
                    # translate errno using _sys_errlist (or similar), but
                    # how can this be done from Python?
                    raise WindowsError(*e.args)
            finally:
                # Child is launched. Close the parent's copy of those pipe
                # handles that only the child should have open.  You need
                # to make sure that no handles to the write end of the
                # output pipe are maintained in this process or else the
                # pipe will not close when the child process exits and the
                # ReadFile will hang.
                if p2cread != -1:
                    p2cread.Close()
                if c2pwrite != -1:
                    c2pwrite.Close()
                if errwrite != -1:
                    errwrite.Close()
                if hasattr(self, '_devnull'):
                    os.close(self._devnull)

            # Retain the process handle, but close the thread handle
            self._child_created = True
            self._handle = hp
            self.pid = pid
            ht.Close()

        def _internal_poll(self, _deadstate=None,
                _WaitForSingleObject=_subprocess.WaitForSingleObject,
                _WAIT_OBJECT_0=_subprocess.WAIT_OBJECT_0,
                _GetExitCodeProcess=_subprocess.GetExitCodeProcess):
            """Check if child process has terminated.  Returns returncode
            attribute.

            This method is called by __del__, so it can only refer to objects
            in its local scope.

            """
            if self.returncode is None:
                if _WaitForSingleObject(self._handle, 0) == _WAIT_OBJECT_0:
                    self.returncode = _GetExitCodeProcess(self._handle)
            return self.returncode


        def wait(self, timeout=None, endtime=None):
            """Wait for child process to terminate.  Returns returncode
            attribute."""
            if endtime is not None:
                timeout_millis = self._remaining_time(endtime)
            if timeout is None:
                timeout_millis = _subprocess.INFINITE
            else:
                timeout_millis = int(timeout * 1000)
            if self.returncode is None:
                result = _subprocess.WaitForSingleObject(self._handle,
                                                         timeout_millis)
                if result == _WAIT_TIMEOUT:
                    raise TimeoutExpired(self.args, timeout)
                self.returncode = _subprocess.GetExitCodeProcess(self._handle)
            return self.returncode


        def _readerthread(self, fh, buffer):
            buffer.append(fh.read())
            fh.close()


        def _communicate(self, input, endtime, orig_timeout):
            # Start reader threads feeding into a list hanging off of this
            # object, unless they've already been started.
            if self.stdout and not hasattr(self, "_stdout_buff"):
                self._stdout_buff = []
                self.stdout_thread = \
                        threading.Thread(target=self._readerthread,
                                         args=(self.stdout, self._stdout_buff))
                self.stdout_thread.daemon = True
                self.stdout_thread.start()
            if self.stderr and not hasattr(self, "_stderr_buff"):
                self._stderr_buff = []
                self.stderr_thread = \
                        threading.Thread(target=self._readerthread,
                                         args=(self.stderr, self._stderr_buff))
                self.stderr_thread.daemon = True
                self.stderr_thread.start()

            if self.stdin:
                self._stdin_write(input)

            # Wait for the reader threads, or time out.  If we time out, the
            # threads remain reading and the fds left open in case the user
            # calls communicate again.
            if self.stdout is not None:
                self.stdout_thread.join(self._remaining_time(endtime))
                if self.stdout_thread.isAlive():
                    raise TimeoutExpired(self.args, orig_timeout)
            if self.stderr is not None:
                self.stderr_thread.join(self._remaining_time(endtime))
                if self.stderr_thread.isAlive():
                    raise TimeoutExpired(self.args, orig_timeout)

            # Collect the output from and close both pipes, now that we know
            # both have been read successfully.
            stdout = None
            stderr = None
            if self.stdout:
                stdout = self._stdout_buff
                self.stdout.close()
            if self.stderr:
                stderr = self._stderr_buff
                self.stderr.close()

            # All data exchanged.  Translate lists into strings.
            if stdout is not None:
                stdout = stdout[0]
            if stderr is not None:
                stderr = stderr[0]

            # Translate newlines, if requested.  We cannot let the file
            # object do the translation: It is based on stdio, which is
            # impossible to combine with select (unless forcing no
            # buffering).
            if self.universal_newlines and hasattr(file, 'newlines'):
                if stdout:
                    stdout = self._translate_newlines(stdout)
                if stderr:
                    stderr = self._translate_newlines(stderr)

            return (stdout, stderr)

        def send_signal(self, sig):
            """Send a signal to the process."""
            # Don't signal a process that we know has already died.
            if self.returncode is not None:
                return
            if sig == signal.SIGTERM:
                self.terminate()
            elif sig == signal.CTRL_C_EVENT:
                os.kill(self.pid, signal.CTRL_C_EVENT)
            elif sig == signal.CTRL_BREAK_EVENT:
                os.kill(self.pid, signal.CTRL_BREAK_EVENT)
            else:
                raise ValueError("Unsupported signal: %s" % sig)

        def terminate(self):
            """Terminates the process."""
            # Don't terminate a process that we know has already died.
            if self.returncode is not None:
                return
            _subprocess.TerminateProcess(self._handle, 1)

        kill = terminate

    else:
        #
        # POSIX methods
        #
        def _get_handles(self, stdin, stdout, stderr):
            """Construct and return tuple with IO objects:
            p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite
            """
            p2cread, p2cwrite = -1, -1
            c2pread, c2pwrite = -1, -1
            errread, errwrite = -1, -1

            if stdin is None:
                pass
            elif stdin == PIPE:
                p2cread, p2cwrite = _create_pipe()
            elif stdin == DEVNULL:
                p2cread = self._get_devnull()
            elif isinstance(stdin, int):
                p2cread = stdin
            else:
                # Assuming file-like object
                p2cread = stdin.fileno()

            if stdout is None:
                pass
            elif stdout == PIPE:
                c2pread, c2pwrite = _create_pipe()
            elif stdout == DEVNULL:
                c2pwrite = self._get_devnull()
            elif isinstance(stdout, int):
                c2pwrite = stdout
            else:
                # Assuming file-like object
                c2pwrite = stdout.fileno()

            if stderr is None:
                pass
            elif stderr == PIPE:
                errread, errwrite = _create_pipe()
            elif stderr == STDOUT:
                if c2pwrite != -1:
                    errwrite = c2pwrite
                else: # child's stdout is not set, use parent's stdout
                    errwrite = sys.__stdout__.fileno()
            elif stderr == DEVNULL:
                errwrite = self._get_devnull()
            elif isinstance(stderr, int):
                errwrite = stderr
            else:
                # Assuming file-like object
                errwrite = stderr.fileno()

            return (p2cread, p2cwrite,
                    c2pread, c2pwrite,
                    errread, errwrite)


        if hasattr(os, 'closerange'):  # Introduced in 2.6
            @staticmethod
            def _closerange(fd_low, fd_high):
                os.closerange(fd_low, fd_high)
        else:
            @staticmethod
            def _closerange(fd_low, fd_high):
                for fd in xrange(fd_low, fd_high):
                    while True:
                        try:
                            os.close(fd)
                        except (OSError, IOError), e:
                            if e.errno == errno.EINTR:
                                continue
                            break


        def _close_fds(self, but):
            self._closerange(3, but)
            self._closerange(but + 1, MAXFD)


        def _close_all_but_a_sorted_few_fds(self, fds_to_keep):
            # precondition: fds_to_keep must be sorted and unique
            start_fd = 3
            for fd in fds_to_keep:
                if fd >= start_fd:
                    self._closerange(start_fd, fd)
                    start_fd = fd + 1
            if start_fd <= MAXFD:
                self._closerange(start_fd, MAXFD)


        def _execute_child(self, args, executable, preexec_fn, close_fds,
                           pass_fds, cwd, env, universal_newlines,
                           startupinfo, creationflags, shell,
                           p2cread, p2cwrite,
                           c2pread, c2pwrite,
                           errread, errwrite,
                           restore_signals, start_new_session):
            """Execute program (POSIX version)"""

            if isinstance(args, types.StringTypes):
                args = [args]
            else:
                args = list(args)

            if shell:
                args = ["/bin/sh", "-c"] + args
                if executable:
                    args[0] = executable

            if executable is None:
                executable = args[0]
            orig_executable = executable

            # For transferring possible exec failure from child to parent.
            # Data format: "exception name:hex errno:description"
            # Pickle is not used; it is complex and involves memory allocation.
            errpipe_read, errpipe_write = _create_pipe()
            try:
                try:

                    if _posixsubprocess:
                        fs_encoding = sys.getfilesystemencoding()
                        def fs_encode(s):
                            """Encode s for use in the env, fs or cmdline."""
                            if isinstance(s, str):
                                return s
                            else:
                                return s.encode(fs_encoding, 'strict')

                        # We must avoid complex work that could involve
                        # malloc or free in the child process to avoid
                        # potential deadlocks, thus we do all this here.
                        # and pass it to fork_exec()

                        if env is not None:
                            env_list = [fs_encode(k) + '=' + fs_encode(v)
                                        for k, v in env.items()]
                        else:
                            env_list = None  # Use execv instead of execve.
                        if os.path.dirname(executable):
                            executable_list = (fs_encode(executable),)
                        else:
                            # This matches the behavior of os._execvpe().
                            path_list = _get_exec_path(env)
                            executable_list = (os.path.join(dir, executable)
                                               for dir in path_list)
                            executable_list = tuple(fs_encode(exe)
                                                    for exe in executable_list)
                        fds_to_keep = set(pass_fds)
                        fds_to_keep.add(errpipe_write)
                        self.pid = _posixsubprocess.fork_exec(
                                args, executable_list,
                                close_fds, sorted(fds_to_keep), cwd, env_list,
                                p2cread, p2cwrite, c2pread, c2pwrite,
                                errread, errwrite,
                                errpipe_read, errpipe_write,
                                restore_signals, start_new_session, preexec_fn)
                        self._child_created = True
                    else:
                        # Pure Python implementation: It is not thread safe.
                        # This implementation may deadlock in the child if your
                        # parent process has any other threads running.

                        gc_was_enabled = gc.isenabled()
                        # Disable gc to avoid bug where gc -> file_dealloc ->
                        # write to stderr -> hang.  See issue1336
                        gc.disable()
                        try:
                            self.pid = os.fork()
                        except:
                            if gc_was_enabled:
                                gc.enable()
                            raise
                        self._child_created = True
                        if self.pid == 0:
                            # Child
                            reached_preexec = False
                            try:
                                # Close parent's pipe ends
                                if p2cwrite != -1:
                                    os.close(p2cwrite)
                                if c2pread != -1:
                                    os.close(c2pread)
                                if errread != -1:
                                    os.close(errread)
                                os.close(errpipe_read)

                                # When duping fds, if there arises a situation
                                # where one of the fds is either 0, 1 or 2, it
                                # is possible that it is overwritten (#12607).
                                if c2pwrite == 0:
                                    c2pwrite = os.dup(c2pwrite)
                                if errwrite == 0 or errwrite == 1:
                                    errwrite = os.dup(errwrite)

                                # Dup fds for child
                                def _dup2(a, b):
                                    # dup2() removes the CLOEXEC flag but
                                    # we must do it ourselves if dup2()
                                    # would be a no-op (issue #10806).
                                    if a == b:
                                        _set_cloexec(a, False)
                                    elif a != -1:
                                        os.dup2(a, b)
                                _dup2(p2cread, 0)
                                _dup2(c2pwrite, 1)
                                _dup2(errwrite, 2)

                                # Close pipe fds.  Make sure we don't close the
                                # same fd more than once, or standard fds.
                                closed = set()
                                for fd in [p2cread, c2pwrite, errwrite]:
                                    if fd > 2 and fd not in closed:
                                        os.close(fd)
                                        closed.add(fd)

                                if cwd is not None:
                                    os.chdir(cwd)

                                # This is a copy of Python/pythonrun.c
                                # _Py_RestoreSignals().  If that were exposed
                                # as a sys._py_restoresignals func it would be
                                # better.. but this pure python implementation
                                # isn't likely to be used much anymore.
                                if restore_signals:
                                    signals = ('SIGPIPE', 'SIGXFZ', 'SIGXFSZ')
                                    for sig in signals:
                                        if hasattr(signal, sig):
                                            signal.signal(getattr(signal, sig),
                                                          signal.SIG_DFL)

                                if start_new_session and hasattr(os, 'setsid'):
                                    os.setsid()

                                reached_preexec = True
                                if preexec_fn:
                                    preexec_fn()

                                # Close all other fds, if asked for - after
                                # preexec_fn(), which may open FDs.
                                if close_fds:
                                    if pass_fds:
                                        fds_to_keep = set(pass_fds)
                                        fds_to_keep.add(errpipe_write)
                                        self._close_all_but_a_sorted_few_fds(
                                                sorted(fds_to_keep))
                                    else:
                                        self._close_fds(but=errpipe_write)

                                if env is None:
                                    os.execvp(executable, args)
                                else:
                                    os.execvpe(executable, args, env)

                            except:
                                try:
                                    exc_type, exc_value = sys.exc_info()[:2]
                                    if isinstance(exc_value, OSError):
                                        errno_num = exc_value.errno
                                    else:
                                        errno_num = 0
                                    if not reached_preexec:
                                        exc_value = "noexec"
                                    message = '%s:%x:%s' % (exc_type.__name__,
                                                            errno_num, exc_value)
                                    os.write(errpipe_write, message)
                                except Exception:
                                    # We MUST not allow anything odd happening
                                    # above to prevent us from exiting below.
                                    pass

                            # This exitcode won't be reported to applications
                            # so it really doesn't matter what we return.
                            os._exit(255)

                        # Parent
                        if gc_was_enabled:
                            gc.enable()
                finally:
                    # be sure the FD is closed no matter what
                    os.close(errpipe_write)

                # A pair of non -1s means we created both fds and are
                # responsible for closing them.
                # self._devnull is not always defined.
                devnull_fd = getattr(self, '_devnull', None)
                if p2cread != -1 and p2cwrite != -1 and p2cread != devnull_fd:
                    os.close(p2cread)
                if c2pwrite != -1 and c2pread != -1 and c2pwrite != devnull_fd:
                    os.close(c2pwrite)
                if errwrite != -1 and errread != -1 and errwrite != devnull_fd:
                    os.close(errwrite)
                if devnull_fd is not None:
                    os.close(devnull_fd)
                # Prevent a double close of these fds from __init__ on error.
                self._closed_child_pipe_fds = True

                # Wait for exec to fail or succeed; possibly raising exception
                # exception (limited in size)
                errpipe_data = ''
                while True:
                    part = _eintr_retry_call(os.read, errpipe_read, 50000)
                    errpipe_data += part
                    if not part or len(errpipe_data) > 50000:
                        break
            finally:
                # be sure the FD is closed no matter what
                os.close(errpipe_read)

            if errpipe_data != "":
                try:
                    _eintr_retry_call(os.waitpid, self.pid, 0)
                except OSError, e:
                    if e.errno != errno.ECHILD:
                        raise
                try:
                    exception_name, hex_errno, err_msg = (
                            errpipe_data.split(':', 2))
                except ValueError:
                    exception_name = 'RuntimeError'
                    hex_errno = '0'
                    err_msg = ('Bad exception data from child: ' +
                               repr(errpipe_data))
                child_exception_type = getattr(
                        exceptions, exception_name, RuntimeError)
                if issubclass(child_exception_type, OSError) and hex_errno:
                    errno_num = int(hex_errno, 16)
                    child_exec_never_called = (err_msg == "noexec")
                    if child_exec_never_called:
                        err_msg = ""
                    if errno_num != 0:
                        err_msg = os.strerror(errno_num)
                        if errno_num == errno.ENOENT:
                            if child_exec_never_called:
                                # The error must be from chdir(cwd).
                                err_msg += ': ' + repr(cwd)
                            else:
                                err_msg += ': ' + repr(orig_executable)
                    raise child_exception_type(errno_num, err_msg)
                try:
                    exception = child_exception_type(err_msg)
                except Exception:
                    exception = RuntimeError(
                            'Could not re-raise %r exception from the'
                            ' child with error message %r' %
                            (child_exception_type, err_msg))
                raise exception


        def _handle_exitstatus(self, sts, _WIFSIGNALED=os.WIFSIGNALED,
                _WTERMSIG=os.WTERMSIG, _WIFEXITED=os.WIFEXITED,
                _WEXITSTATUS=os.WEXITSTATUS, _WIFSTOPPED=os.WIFSTOPPED,
                _WSTOPSIG=os.WSTOPSIG):
            """All callers to this function MUST hold self._waitpid_lock."""
            # This method is called (indirectly) by __del__, so it cannot
            # refer to anything outside of its local scope."""
            if _WIFSIGNALED(sts):
                self.returncode = -_WTERMSIG(sts)
            elif _WIFEXITED(sts):
                self.returncode = _WEXITSTATUS(sts)
            elif _WIFSTOPPED(sts):
                self.returncode = -_WSTOPSIG(sts)
            else:
                # Should never happen
                raise RuntimeError("Unknown child exit status!")


        def _internal_poll(self, _deadstate=None, _waitpid=os.waitpid,
                _WNOHANG=os.WNOHANG, _os_error=os.error, _ECHILD=errno.ECHILD):
            """Check if child process has terminated.  Returns returncode
            attribute.

            This method is called by __del__, so it cannot reference anything
            outside of the local scope (nor can any methods it calls).

            """
            if self.returncode is None:
                if not self._waitpid_lock.acquire(False):
                    # Something else is busy calling waitpid.  Don't allow two
                    # at once.  We know nothing yet.
                    return None
                try:
                    try:
                        if self.returncode is not None:
                            return self.returncode  # Another thread waited.
                        pid, sts = _waitpid(self.pid, _WNOHANG)
                        if pid == self.pid:
                            self._handle_exitstatus(sts)
                    except _os_error, e:
                        if _deadstate is not None:
                            self.returncode = _deadstate
                        elif e.errno == _ECHILD:
                            # This happens if SIGCLD is set to be ignored or
                            # waiting for child processes has otherwise been
                            # disabled for our process.  This child is dead, we
                            # can't get the status.
                            # http://bugs.python.org/issue15756
                            self.returncode = 0
                finally:
                    self._waitpid_lock.release()
            return self.returncode


        def _try_wait(self, wait_flags):
            """All callers to this function MUST hold self._waitpid_lock."""
            try:
                (pid, sts) = _eintr_retry_call(os.waitpid, self.pid, wait_flags)
            except OSError, e:
                if e.errno != errno.ECHILD:
                    raise
                # This happens if SIGCLD is set to be ignored or waiting
                # for child processes has otherwise been disabled for our
                # process.  This child is dead, we can't get the status.
                pid = self.pid
                sts = 0
            return (pid, sts)


        def wait(self, timeout=None, endtime=None):
            """Wait for child process to terminate.  Returns returncode
            attribute."""
            if self.returncode is not None:
                return self.returncode

            # endtime is preferred to timeout.  timeout is only used for
            # printing.
            if endtime is not None or timeout is not None:
                if endtime is None:
                    endtime = time.time() + timeout
                elif timeout is None:
                    timeout = self._remaining_time(endtime)

            if endtime is not None:
                # Enter a busy loop if we have a timeout.  This busy loop was
                # cribbed from Lib/threading.py in Thread.wait() at r71065.
                delay = 0.0005 # 500 us -> initial delay of 1 ms
                while True:
                    if self._waitpid_lock.acquire(False):
                        try:
                            if self.returncode is not None:
                                break  # Another thread waited.
                            (pid, sts) = self._try_wait(os.WNOHANG)
                            assert pid == self.pid or pid == 0
                            if pid == self.pid:
                                self._handle_exitstatus(sts)
                                break
                        finally:
                            self._waitpid_lock.release()
                    remaining = self._remaining_time(endtime)
                    if remaining <= 0:
                        raise TimeoutExpired(self.args, timeout)
                    delay = min(delay * 2, remaining, .05)
                    time.sleep(delay)
            else:
                while self.returncode is None:
                    self._waitpid_lock.acquire()
                    try:
                        if self.returncode is not None:
                            break  # Another thread waited.
                        (pid, sts) = self._try_wait(0)
                        # Check the pid and loop as waitpid has been known to
                        # return 0 even without WNOHANG in odd situations.
                        # http://bugs.python.org/issue14396.
                        if pid == self.pid:
                            self._handle_exitstatus(sts)
                    finally:
                        self._waitpid_lock.release()
            return self.returncode


        def _communicate(self, input, endtime, orig_timeout):
            if self.stdin and not self._communication_started:
                # Flush stdio buffer.  This might block, if the user has
                # been writing to .stdin in an uncontrolled fashion.
                self.stdin.flush()
                if not input:
                    self.stdin.close()

            if _has_poll:
                stdout, stderr = self._communicate_with_poll(input, endtime,
                                                             orig_timeout)
            else:
                stdout, stderr = self._communicate_with_select(input, endtime,
                                                               orig_timeout)

            self.wait(timeout=self._remaining_time(endtime))

            # All data exchanged.  Translate lists into strings.
            if stdout is not None:
                stdout = ''.join(stdout)
            if stderr is not None:
                stderr = ''.join(stderr)

            # Translate newlines, if requested.  We cannot let the file
            # object do the translation: It is based on stdio, which is
            # impossible to combine with select (unless forcing no
            # buffering).
            if self.universal_newlines and hasattr(file, 'newlines'):
                if stdout:
                    stdout = self._translate_newlines(stdout)
                if stderr:
                    stderr = self._translate_newlines(stderr)

            return (stdout, stderr)


        def _communicate_with_poll(self, input, endtime, orig_timeout):
            stdout = None # Return
            stderr = None # Return

            if not self._communication_started:
                self._fd2file = {}

            poller = select.poll()
            def register_and_append(file_obj, eventmask):
                poller.register(file_obj.fileno(), eventmask)
                self._fd2file[file_obj.fileno()] = file_obj

            def close_unregister_and_remove(fd):
                poller.unregister(fd)
                self._fd2file[fd].close()
                self._fd2file.pop(fd)

            if self.stdin and input:
                register_and_append(self.stdin, select.POLLOUT)

            # Only create this mapping if we haven't already.
            if not self._communication_started:
                self._fd2output = {}
                if self.stdout:
                    self._fd2output[self.stdout.fileno()] = []
                if self.stderr:
                    self._fd2output[self.stderr.fileno()] = []

            select_POLLIN_POLLPRI = select.POLLIN | select.POLLPRI
            if self.stdout:
                register_and_append(self.stdout, select_POLLIN_POLLPRI)
                stdout = self._fd2output[self.stdout.fileno()]
            if self.stderr:
                register_and_append(self.stderr, select_POLLIN_POLLPRI)
                stderr = self._fd2output[self.stderr.fileno()]

            # Save the input here so that if we time out while communicating,
            # we can continue sending input if we retry.
            if self.stdin and self._input is None:
                self._input_offset = 0
                self._input = input
                if self.universal_newlines and isinstance(self._input, unicode):
                    self._input = self._input.encode(
                            self.stdin.encoding or sys.getdefaultencoding())

            while self._fd2file:
                try:
                    ready = poller.poll(self._remaining_time(endtime))
                except select.error, e:
                    if e.args[0] == errno.EINTR:
                        continue
                    raise
                self._check_timeout(endtime, orig_timeout)

                for fd, mode in ready:
                    if mode & select.POLLOUT:
                        chunk = self._input[self._input_offset :
                                            self._input_offset + _PIPE_BUF]
                        self._input_offset += os.write(fd, chunk)
                        if self._input_offset >= len(self._input):
                            close_unregister_and_remove(fd)
                    elif mode & select_POLLIN_POLLPRI:
                        data = os.read(fd, 4096)
                        if not data:
                            close_unregister_and_remove(fd)
                        self._fd2output[fd].append(data)
                    else:
                        # Ignore hang up or errors.
                        close_unregister_and_remove(fd)

            return (stdout, stderr)


        def _communicate_with_select(self, input, endtime, orig_timeout):
            if not self._communication_started:
                self._read_set = []
                self._write_set = []
                if self.stdin and input:
                    self._write_set.append(self.stdin)
                if self.stdout:
                    self._read_set.append(self.stdout)
                if self.stderr:
                    self._read_set.append(self.stderr)

            if self.stdin and self._input is None:
                self._input_offset = 0
                self._input = input
                if self.universal_newlines and isinstance(self._input, unicode):
                    self._input = self._input.encode(
                            self.stdin.encoding or sys.getdefaultencoding())

            stdout = None # Return
            stderr = None # Return

            if self.stdout:
                if not self._communication_started:
                    self._stdout_buff = []
                stdout = self._stdout_buff
            if self.stderr:
                if not self._communication_started:
                    self._stderr_buff = []
                stderr = self._stderr_buff

            while self._read_set or self._write_set:
                try:
                    (rlist, wlist, xlist) = \
                        select.select(self._read_set, self._write_set, [],
                                      self._remaining_time(endtime))
                except select.error, e:
                    if e.args[0] == errno.EINTR:
                        continue
                    raise

                # According to the docs, returning three empty lists indicates
                # that the timeout expired.
                if not (rlist or wlist or xlist):
                    raise TimeoutExpired(self.args, orig_timeout)
                # We also check what time it is ourselves for good measure.
                self._check_timeout(endtime, orig_timeout)

                if self.stdin in wlist:
                    chunk = self._input[self._input_offset :
                                        self._input_offset + _PIPE_BUF]
                    try:
                        bytes_written = os.write(self.stdin.fileno(), chunk)
                    except EnvironmentError as e:
                        if e.errno == errno.EPIPE:
                            self._write_set.remove(self.stdin)
                            self.stdin.close()
                        else:
                            raise
                    else:
                        self._input_offset += bytes_written
                        if self._input_offset >= len(self._input):
                            self.stdin.close()
                            self._write_set.remove(self.stdin)

                if self.stdout in rlist:
                    data = os.read(self.stdout.fileno(), 1024)
                    if data == "":
                        self.stdout.close()
                        self._read_set.remove(self.stdout)
                    stdout.append(data)

                if self.stderr in rlist:
                    data = os.read(self.stderr.fileno(), 1024)
                    if data == "":
                        self.stderr.close()
                        self._read_set.remove(self.stderr)
                    stderr.append(data)

            return (stdout, stderr)


        def send_signal(self, sig):
            """Send a signal to the process
            """
            # Skip signalling a process that we know has already died.
            if self.returncode is None:
                os.kill(self.pid, sig)

        def terminate(self):
            """Terminate the process with SIGTERM
            """
            self.send_signal(signal.SIGTERM)

        def kill(self):
            """Kill the process with SIGKILL
            """
            self.send_signal(signal.SIGKILL)
