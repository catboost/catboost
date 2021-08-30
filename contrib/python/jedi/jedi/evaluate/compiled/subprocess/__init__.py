"""
Makes it possible to do the compiled analysis in a subprocess. This has two
goals:

1. Making it safer - Segfaults and RuntimeErrors as well as stdout/stderr can
   be ignored and dealt with.
2. Make it possible to handle different Python versions as well as virtualenvs.
"""

import os
import sys
import subprocess
import socket
import errno
import weakref
import traceback
from functools import partial
from threading import Thread
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty  # python 2.7

from jedi._compatibility import queue, is_py3, force_unicode, \
    pickle_dump, pickle_load, GeneralizedPopen, print_to_stderr
from jedi import debug
from jedi.cache import memoize_method
from jedi.evaluate.compiled.subprocess import functions
from jedi.evaluate.compiled.access import DirectObjectAccess, AccessPath, \
    SignatureParam
from jedi.api.exceptions import InternalError


_MAIN_PATH = os.path.join(os.path.dirname(__file__), '__main__.py')


def _enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


def _add_stderr_to_debug(stderr_queue):
    while True:
        # Try to do some error reporting from the subprocess and print its
        # stderr contents.
        try:
            line = stderr_queue.get_nowait()
            line = line.decode('utf-8', 'replace')
            debug.warning('stderr output: %s' % line.rstrip('\n'))
        except Empty:
            break


def _get_function(name):
    return getattr(functions, name)


class _EvaluatorProcess(object):
    def __init__(self, evaluator):
        self._evaluator_weakref = weakref.ref(evaluator)
        self._evaluator_id = id(evaluator)
        self._handles = {}

    def get_or_create_access_handle(self, obj):
        id_ = id(obj)
        try:
            return self.get_access_handle(id_)
        except KeyError:
            access = DirectObjectAccess(self._evaluator_weakref(), obj)
            handle = AccessHandle(self, access, id_)
            self.set_access_handle(handle)
            return handle

    def get_access_handle(self, id_):
        return self._handles[id_]

    def set_access_handle(self, handle):
        self._handles[handle.id] = handle


class EvaluatorSameProcess(_EvaluatorProcess):
    """
    Basically just an easy access to functions.py. It has the same API
    as EvaluatorSubprocess and does the same thing without using a subprocess.
    This is necessary for the Interpreter process.
    """
    def __getattr__(self, name):
        return partial(_get_function(name), self._evaluator_weakref())


class EvaluatorSubprocess(_EvaluatorProcess):
    def __init__(self, evaluator, compiled_subprocess):
        super(EvaluatorSubprocess, self).__init__(evaluator)
        self._used = False
        self._compiled_subprocess = compiled_subprocess

    def __getattr__(self, name):
        func = _get_function(name)

        def wrapper(*args, **kwargs):
            self._used = True

            result = self._compiled_subprocess.run(
                self._evaluator_weakref(),
                func,
                args=args,
                kwargs=kwargs,
            )
            # IMO it should be possible to create a hook in pickle.load to
            # mess with the loaded objects. However it's extremely complicated
            # to work around this so just do it with this call. ~ dave
            return self._convert_access_handles(result)

        return wrapper

    def _convert_access_handles(self, obj):
        if isinstance(obj, SignatureParam):
            return SignatureParam(*self._convert_access_handles(tuple(obj)))
        elif isinstance(obj, tuple):
            return tuple(self._convert_access_handles(o) for o in obj)
        elif isinstance(obj, list):
            return [self._convert_access_handles(o) for o in obj]
        elif isinstance(obj, AccessHandle):
            try:
                # Rewrite the access handle to one we're already having.
                obj = self.get_access_handle(obj.id)
            except KeyError:
                obj.add_subprocess(self)
                self.set_access_handle(obj)
        elif isinstance(obj, AccessPath):
            return AccessPath(self._convert_access_handles(obj.accesses))
        return obj

    def __del__(self):
        if self._used and not self._compiled_subprocess.is_crashed:
            self._compiled_subprocess.delete_evaluator(self._evaluator_id)


class CompiledSubprocess(object):
    is_crashed = False
    # Start with 2, gets set after _get_info.
    _pickle_protocol = 2

    def __init__(self, executable):
        self._executable = executable
        self._evaluator_deletion_queue = queue.deque()

    def __repr__(self):
        pid = os.getpid()
        return '<%s _executable=%r, _pickle_protocol=%r, is_crashed=%r, pid=%r>' % (
            self.__class__.__name__,
            self._executable,
            self._pickle_protocol,
            self.is_crashed,
            pid,
        )

    @property
    @memoize_method
    def _process(self):
        debug.dbg('Start environment subprocess %s', self._executable)
        parso_path = sys.modules['parso'].__file__
        args = (
            self._executable,
            _MAIN_PATH,
            os.path.dirname(os.path.dirname(parso_path)),
            '.'.join(str(x) for x in sys.version_info[:3]),
        )
        process = GeneralizedPopen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Use system default buffering on Python 2 to improve performance
            # (this is already the case on Python 3).
            bufsize=-1
        )
        self._stderr_queue = Queue()
        self._stderr_thread = t = Thread(
            target=_enqueue_output,
            args=(process.stderr, self._stderr_queue)
        )
        t.daemon = True
        t.start()
        return process

    def run(self, evaluator, function, args=(), kwargs={}):
        # Delete old evaluators.
        while True:
            try:
                evaluator_id = self._evaluator_deletion_queue.pop()
            except IndexError:
                break
            else:
                self._send(evaluator_id, None)

        assert callable(function)
        return self._send(id(evaluator), function, args, kwargs)

    def get_sys_path(self):
        return self._send(None, functions.get_sys_path, (), {})

    def _kill(self):
        self.is_crashed = True
        try:
            self._process.kill()
            self._process.wait()
        except (AttributeError, TypeError):
            # If the Python process is terminating, it will remove some modules
            # earlier than others and in general it's unclear how to deal with
            # that so we just ignore the exceptions here.
            pass

    def __del__(self):
        if not self.is_crashed:
            self._kill()

    def _send(self, evaluator_id, function, args=(), kwargs={}):
        if self.is_crashed:
            raise InternalError("The subprocess %s has crashed." % self._executable)

        if not is_py3:
            # Python 2 compatibility
            kwargs = {force_unicode(key): value for key, value in kwargs.items()}

        data = evaluator_id, function, args, kwargs
        try:
            pickle_dump(data, self._process.stdin, self._pickle_protocol)
        except (socket.error, IOError) as e:
            # Once Python2 will be removed we can just use `BrokenPipeError`.
            # Also, somehow in windows it returns EINVAL instead of EPIPE if
            # the subprocess dies.
            if e.errno not in (errno.EPIPE, errno.EINVAL):
                # Not a broken pipe
                raise
            self._kill()
            raise InternalError("The subprocess %s was killed. Maybe out of memory?"
                                % self._executable)

        try:
            is_exception, traceback, result = pickle_load(self._process.stdout)
        except EOFError as eof_error:
            try:
                stderr = self._process.stderr.read().decode('utf-8', 'replace')
            except Exception as exc:
                stderr = '<empty/not available (%r)>' % exc
            self._kill()
            _add_stderr_to_debug(self._stderr_queue)
            raise InternalError(
                "The subprocess %s has crashed (%r, stderr=%s)." % (
                    self._executable,
                    eof_error,
                    stderr,
                ))

        _add_stderr_to_debug(self._stderr_queue)

        if is_exception:
            # Replace the attribute error message with a the traceback. It's
            # way more informative.
            result.args = (traceback,)
            raise result
        return result

    def delete_evaluator(self, evaluator_id):
        """
        Currently we are not deleting evalutors instantly. They only get
        deleted once the subprocess is used again. It would probably a better
        solution to move all of this into a thread. However, the memory usage
        of a single evaluator shouldn't be that high.
        """
        # With an argument - the evaluator gets deleted.
        self._evaluator_deletion_queue.append(evaluator_id)


class Listener(object):
    def __init__(self, pickle_protocol):
        self._evaluators = {}
        # TODO refactor so we don't need to process anymore just handle
        # controlling.
        self._process = _EvaluatorProcess(Listener)
        self._pickle_protocol = pickle_protocol

    def _get_evaluator(self, function, evaluator_id):
        from jedi.evaluate import Evaluator

        try:
            evaluator = self._evaluators[evaluator_id]
        except KeyError:
            from jedi.api.environment import InterpreterEnvironment
            evaluator = Evaluator(
                # The project is not actually needed. Nothing should need to
                # access it.
                project=None,
                environment=InterpreterEnvironment()
            )
            self._evaluators[evaluator_id] = evaluator
        return evaluator

    def _run(self, evaluator_id, function, args, kwargs):
        if evaluator_id is None:
            return function(*args, **kwargs)
        elif function is None:
            del self._evaluators[evaluator_id]
        else:
            evaluator = self._get_evaluator(function, evaluator_id)

            # Exchange all handles
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, AccessHandle):
                    args[i] = evaluator.compiled_subprocess.get_access_handle(arg.id)
            for key, value in kwargs.items():
                if isinstance(value, AccessHandle):
                    kwargs[key] = evaluator.compiled_subprocess.get_access_handle(value.id)

            return function(evaluator, *args, **kwargs)

    def listen(self):
        stdout = sys.stdout
        # Mute stdout. Nobody should actually be able to write to it,
        # because stdout is used for IPC.
        sys.stdout = open(os.devnull, 'w')
        stdin = sys.stdin
        if sys.version_info[0] > 2:
            stdout = stdout.buffer
            stdin = stdin.buffer
        # Python 2 opens streams in text mode on Windows. Set stdout and stdin
        # to binary mode.
        elif sys.platform == 'win32':
            import msvcrt
            msvcrt.setmode(stdout.fileno(), os.O_BINARY)
            msvcrt.setmode(stdin.fileno(), os.O_BINARY)

        while True:
            try:
                payload = pickle_load(stdin)
            except EOFError:
                # It looks like the parent process closed.
                # Don't make a big fuss here and just exit.
                exit(0)
            try:
                result = False, None, self._run(*payload)
            except Exception as e:
                result = True, traceback.format_exc(), e

            pickle_dump(result, stdout, self._pickle_protocol)


class AccessHandle(object):
    def __init__(self, subprocess, access, id_):
        self.access = access
        self._subprocess = subprocess
        self.id = id_

    def add_subprocess(self, subprocess):
        self._subprocess = subprocess

    def __repr__(self):
        try:
            detail = self.access
        except AttributeError:
            detail = '#' + str(self.id)
        return '<%s of %s>' % (self.__class__.__name__, detail)

    def __getstate__(self):
        return self.id

    def __setstate__(self, state):
        self.id = state

    def __getattr__(self, name):
        if name in ('id', 'access') or name.startswith('_'):
            raise AttributeError("Something went wrong with unpickling")

        #if not is_py3: print >> sys.stderr, name
        #print('getattr', name, file=sys.stderr)
        return partial(self._workaround, force_unicode(name))

    def _workaround(self, name, *args, **kwargs):
        """
        TODO Currently we're passing slice objects around. This should not
        happen. They are also the only unhashable objects that we're passing
        around.
        """
        if args and isinstance(args[0], slice):
            return self._subprocess.get_compiled_method_return(self.id, name, *args, **kwargs)
        return self._cached_results(name, *args, **kwargs)

    @memoize_method
    def _cached_results(self, name, *args, **kwargs):
        #if type(self._subprocess) == EvaluatorSubprocess:
            #print(name, args, kwargs,
                #self._subprocess.get_compiled_method_return(self.id, name, *args, **kwargs)
            #)
        return self._subprocess.get_compiled_method_return(self.id, name, *args, **kwargs)
