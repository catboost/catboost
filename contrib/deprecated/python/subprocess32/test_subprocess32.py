import unittest
from test import test_support
import subprocess32
subprocess = subprocess32
import sys
try:
    import ctypes
except ImportError:
    ctypes = None
else:
    import ctypes.util
import signal
import os
import errno
import tempfile
import textwrap
import time
try:
    import threading
except ImportError:
    threading = None
import re
#import sysconfig
import select
import shutil
try:
    import gc
except ImportError:
    gc = None
import pickle

mswindows = (sys.platform == "win32")
yenv = '''
import os
os.environ['Y_PYTHON_ENTRY_POINT'] = ':main'
'''

#
# Depends on the following external programs: Python
#

if mswindows:
    SETBINARY = ('import msvcrt; msvcrt.setmode(sys.stdout.fileno(), '
                                                'os.O_BINARY);')
else:
    SETBINARY = ''


try:
    mkstemp = tempfile.mkstemp
except AttributeError:
    # tempfile.mkstemp is not available
    def mkstemp():
        """Replacement for mkstemp, calling mktemp."""
        fname = tempfile.mktemp()
        return os.open(fname, os.O_RDWR|os.O_CREAT), fname

try:
    strip_python_stderr = test_support.strip_python_stderr
except AttributeError:
    # Copied from the test.test_support module in 2.7.
    def strip_python_stderr(stderr):
        """Strip the stderr of a Python process from potential debug output
        emitted by the interpreter.

        This will typically be run on the result of the communicate() method
        of a subprocess.Popen object.
        """
        stderr = re.sub(r"\[\d+ refs\]\r?\n?$", "", stderr).strip()
        return stderr

class BaseTestCase(unittest.TestCase):
    def setUp(self):
        os.environ['Y_PYTHON_ENTRY_POINT'] = ':main'
        # Try to minimize the number of children we have so this test
        # doesn't crash on some buildbots (Alphas in particular).
        reap_children()
        if not hasattr(unittest.TestCase, 'addCleanup'):
            self._cleanups = []

    def tearDown(self):
        try:
            for inst in subprocess._active:
                inst.wait()
            subprocess._cleanup()
            self.assertFalse(subprocess._active, "subprocess._active not empty")
        finally:
            if self._use_our_own_cleanup_implementation:
                self._doCleanups()

    if not hasattr(unittest.TestCase, 'assertIn'):
        def assertIn(self, a, b, msg=None):
            self.assert_((a in b), msg or ('%r not in %r' % (a, b)))
        def assertNotIn(self, a, b, msg=None):
            self.assert_((a not in b), msg or ('%r in %r' % (a, b)))

    if not hasattr(unittest.TestCase, 'skipTest'):
        def skipTest(self, message):
            """These will still fail but it'll be clear that it is okay."""
            self.fail('SKIPPED - %s\n' % (message,))

    def _addCleanup(self, function, *args, **kwargs):
        """Add a function, with arguments, to be called when the test is
        completed. Functions added are called on a LIFO basis and are
        called after tearDown on test failure or success.

        Unlike unittest2 or python 2.7, cleanups are not if setUp fails.
        That is easier to implement in this subclass and is all we need.
        """
        self._cleanups.append((function, args, kwargs))

    def _doCleanups(self):
        """Execute all cleanup functions. Normally called for you after
        tearDown."""
        while self._cleanups:
            function, args, kwargs = self._cleanups.pop()
            try:
                function(*args, **kwargs)
            except KeyboardInterrupt:
                raise
            except:
                pass

    _use_our_own_cleanup_implementation = False
    if not hasattr(unittest.TestCase, 'addCleanup'):
        _use_our_own_cleanup_implementation = True
        addCleanup = _addCleanup

    def assertStderrEqual(self, stderr, expected, msg=None):
        # In a debug build, stuff like "[6580 refs]" is printed to stderr at
        # shutdown time.  That frustrates tests trying to check stderr produced
        # from a spawned Python process.
        actual = strip_python_stderr(stderr)
        # strip_python_stderr also strips whitespace, so we do too.
        expected = expected.strip()
        self.assertEqual(actual, expected, msg)


class PopenTestException(Exception):
    pass


class PopenExecuteChildRaises(subprocess32.Popen):
    """Popen subclass for testing cleanup of subprocess.PIPE filehandles when
    _execute_child fails.
    """
    def _execute_child(self, *args, **kwargs):
        raise PopenTestException("Forced Exception for Test")


class ProcessTestCase(BaseTestCase):

    def test_call_seq(self):
        # call() function with sequence argument
        rc = subprocess.call([sys.executable, "-c", yenv +
                              "import sys; sys.exit(47)"])
        self.assertEqual(rc, 47)

    def test_call_timeout(self):
        # call() function with timeout argument; we want to test that the child
        # process gets killed when the timeout expires.  If the child isn't
        # killed, this call will deadlock since subprocess.call waits for the
        # child.
        self.assertRaises(subprocess.TimeoutExpired, subprocess.call,
                          [sys.executable, "-c", yenv + "while True: pass"],
                          timeout=0.1)

    def test_check_call_zero(self):
        # check_call() function with zero return code
        rc = subprocess.check_call([sys.executable, "-c", yenv +
                                    "import sys; sys.exit(0)"])
        self.assertEqual(rc, 0)

    def test_check_call_nonzero(self):
        # check_call() function with non-zero return code
        try:
            subprocess.check_call([sys.executable, "-c", yenv +
                                   "import sys; sys.exit(47)"])
        except subprocess.CalledProcessError, c:
            self.assertEqual(c.returncode, 47)

    def test_check_output(self):
        # check_output() function with zero return code
        output = subprocess.check_output(
                [sys.executable, "-c", yenv + "print 'BDFL'"])
        self.assertIn('BDFL', output)

    def test_check_output_nonzero(self):
        # check_call() function with non-zero return code
        try:
            subprocess.check_output(
                    [sys.executable, "-c", yenv + "import sys; sys.exit(5)"])
        except subprocess.CalledProcessError, c:
            self.assertEqual(c.returncode, 5)

    def test_check_output_stderr(self):
        # check_output() function stderr redirected to stdout
        output = subprocess.check_output(
                [sys.executable, "-c", yenv + "import sys; sys.stderr.write('BDFL')"],
                stderr=subprocess.STDOUT)
        self.assertIn('BDFL', output)

    def test_check_output_stdout_arg(self):
        # check_output() function stderr redirected to stdout
        try:
            output = subprocess.check_output(
                    [sys.executable, "-c", yenv + "print 'will not be run'"],
                    stdout=sys.stdout)
            self.fail("Expected ValueError when stdout arg supplied.")
        except ValueError, c:
            self.assertIn('stdout', c.args[0])

    def test_check_output_timeout(self):
        # check_output() function with timeout arg
        try:
            output = subprocess.check_output(
                    [sys.executable, "-c", yenv +
                     "import sys; sys.stdout.write('BDFL')\n"
                     "sys.stdout.flush()\n"
                     "while True: pass"],
                    timeout=0.5)
        except subprocess.TimeoutExpired, exception:
            self.assertEqual(exception.output, 'BDFL')
        else:
            self.fail("Expected TimeoutExpired.")

    def test_call_kwargs(self):
        # call() function with keyword args
        newenv = os.environ.copy()
        newenv["FRUIT"] = "banana"
        rc = subprocess.call([sys.executable, "-c", yenv +
                              'import sys, os;'
                              'sys.exit(os.getenv("FRUIT")=="banana")'],
                             env=newenv)
        self.assertEqual(rc, 1)

    def test_stdin_none(self):
        # .stdin is None when not redirected
        p = subprocess.Popen([sys.executable, "-c", yenv + 'print "banana"'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        self.assertEqual(p.stdin, None)

    def test_stdout_none(self):
        # .stdout is None when not redirected, and the child's stdout will
        # be inherited from the parent.  In order to test this we run a
        # subprocess in a subprocess:
        # this_test
        #   \-- subprocess created by this test (parent)
        #          \-- subprocess created by the parent subprocess (child)
        # The parent doesn't specify stdout, so the child will use the
        # parent's stdout.  This test checks that the message printed by the
        # child goes to the parent stdout.  The parent also checks that the
        # child's stdout is None.  See #11963.
        code = ('import sys; from subprocess32 import Popen, PIPE;'
                'p = Popen([sys.executable, "-c", "print \'test_stdout_none\'"],'
                '          stdin=PIPE, stderr=PIPE);'
                'p.wait(); assert p.stdout is None;')
        p = subprocess.Popen([sys.executable, "-c", yenv + code],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(p.stdout.close)
        self.addCleanup(p.stderr.close)
        out, err = p.communicate()
        self.assertEqual(p.returncode, 0, err)
        self.assertEqual(out.rstrip(), 'test_stdout_none')

    def test_stderr_none(self):
        # .stderr is None when not redirected
        p = subprocess.Popen([sys.executable, "-c", yenv + 'print "banana"'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        p.wait()
        self.assertEqual(p.stderr, None)

    # For use in the test_cwd* tests below.
    def _normalize_cwd(self, cwd):
        # Normalize an expected cwd (for Tru64 support).
        # We can't use os.path.realpath since it doesn't expand Tru64 {memb}
        # strings.  See bug #1063571.
        original_cwd = os.getcwd()
        os.chdir(cwd)
        cwd = os.getcwd()
        os.chdir(original_cwd)
        return cwd

    # For use in the test_cwd* tests below.
    def _split_python_path(self):
        # Return normalized (python_dir, python_base).
        python_path = os.path.realpath(sys.executable)
        return os.path.split(python_path)

    # For use in the test_cwd* tests below.
    def _assert_cwd(self, expected_cwd, python_arg, **kwargs):
        # Invoke Python via Popen, and assert that (1) the call succeeds,
        # and that (2) the current working directory of the child process
        # matches *expected_cwd*.
        p = subprocess.Popen([python_arg, "-c", yenv +
                              "import os, sys; "
                              "sys.stdout.write(os.getcwd()); "
                              "sys.exit(47)"],
                              stdout=subprocess.PIPE,
                              **kwargs)
        self.addCleanup(p.stdout.close)
        p.wait()
        self.assertEqual(47, p.returncode)
        normcase = os.path.normcase
        self.assertEqual(normcase(expected_cwd),
                         normcase(p.stdout.read().decode("utf-8")))

    def test_cwd(self):
        # Check that cwd changes the cwd for the child process.
        temp_dir = tempfile.gettempdir()
        temp_dir = self._normalize_cwd(temp_dir)
        self._assert_cwd(temp_dir, sys.executable, cwd=temp_dir)

    if not mswindows:  # pending resolution of issue #15533
        def test_cwd_with_relative_arg(self):
            # Check that Popen looks for args[0] relative to cwd if args[0]
            # is relative.
            python_dir, python_base = self._split_python_path()
            rel_python = os.path.join(os.curdir, python_base)

            path = 'tempcwd'
            saved_dir = os.getcwd()
            os.mkdir(path)
            try:
                os.chdir(path)
                wrong_dir = os.getcwd()
                # Before calling with the correct cwd, confirm that the call fails
                # without cwd and with the wrong cwd.
                self.assertRaises(OSError, subprocess.Popen,
                                  [rel_python])
                self.assertRaises(OSError, subprocess.Popen,
                                  [rel_python], cwd=wrong_dir)
                python_dir = self._normalize_cwd(python_dir)
                self._assert_cwd(python_dir, rel_python, cwd=python_dir)
            finally:
                os.chdir(saved_dir)
                shutil.rmtree(path)

        def test_cwd_with_relative_executable(self):
            # Check that Popen looks for executable relative to cwd if executable
            # is relative (and that executable takes precedence over args[0]).
            python_dir, python_base = self._split_python_path()
            rel_python = os.path.join(os.curdir, python_base)
            doesntexist = "somethingyoudonthave"

            path = 'tempcwd'
            saved_dir = os.getcwd()
            os.mkdir(path)
            try:
                os.chdir(path)
                wrong_dir = os.getcwd()
                # Before calling with the correct cwd, confirm that the call fails
                # without cwd and with the wrong cwd.
                self.assertRaises(OSError, subprocess.Popen,
                                  [doesntexist], executable=rel_python)
                self.assertRaises(OSError, subprocess.Popen,
                                  [doesntexist], executable=rel_python,
                                  cwd=wrong_dir)
                python_dir = self._normalize_cwd(python_dir)
                self._assert_cwd(python_dir, doesntexist, executable=rel_python,
                                 cwd=python_dir)
            finally:
                os.chdir(saved_dir)
                shutil.rmtree(path)

    def test_cwd_with_absolute_arg(self):
        # Check that Popen can find the executable when the cwd is wrong
        # if args[0] is an absolute path.
        python_dir, python_base = self._split_python_path()
        abs_python = os.path.join(python_dir, python_base)
        rel_python = os.path.join(os.curdir, python_base)
        wrong_dir = tempfile.mkdtemp()
        wrong_dir = os.path.realpath(wrong_dir)
        try:
            # Before calling with an absolute path, confirm that using a
            # relative path fails.
            self.assertRaises(OSError, subprocess.Popen,
                              [rel_python], cwd=wrong_dir)
            wrong_dir = self._normalize_cwd(wrong_dir)
            self._assert_cwd(wrong_dir, abs_python, cwd=wrong_dir)
        finally:
            shutil.rmtree(wrong_dir)

    def test_executable_with_cwd(self):
        python_dir, python_base = self._split_python_path()
        python_dir = self._normalize_cwd(python_dir)
        self._assert_cwd(python_dir, "somethingyoudonthave",
                         executable=sys.executable, cwd=python_dir)

    #@unittest.skipIf(sysconfig.is_python_build(),
    #                 "need an installed Python. See #7774")
    #def test_executable_without_cwd(self):
    #    # For a normal installation, it should work without 'cwd'
    #    # argument.  For test runs in the build directory, see #7774.
    #    self._assert_cwd('', "somethingyoudonthave", executable=sys.executable)

    def test_stdin_pipe(self):
        # stdin redirection
        p = subprocess.Popen([sys.executable, "-c", yenv +
                         'import sys; sys.exit(sys.stdin.read() == "pear")'],
                        stdin=subprocess.PIPE)
        p.stdin.write("pear")
        p.stdin.close()
        p.wait()
        self.assertEqual(p.returncode, 1)

    def test_stdin_filedes(self):
        # stdin is set to open file descriptor
        tf = tempfile.TemporaryFile()
        d = tf.fileno()
        os.write(d, "pear")
        os.lseek(d, 0, 0)
        p = subprocess.Popen([sys.executable, "-c", yenv +
                         'import sys; sys.exit(sys.stdin.read() == "pear")'],
                         stdin=d)
        p.wait()
        self.assertEqual(p.returncode, 1)

    def test_stdin_fileobj(self):
        # stdin is set to open file object
        tf = tempfile.TemporaryFile()
        tf.write("pear")
        tf.seek(0)
        p = subprocess.Popen([sys.executable, "-c", yenv +
                         'import sys; sys.exit(sys.stdin.read() == "pear")'],
                         stdin=tf)
        p.wait()
        self.assertEqual(p.returncode, 1)

    def test_stdout_pipe(self):
        # stdout redirection
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys; sys.stdout.write("orange")'],
                         stdout=subprocess.PIPE)
        self.assertEqual(p.stdout.read(), "orange")

    def test_stdout_filedes(self):
        # stdout is set to open file descriptor
        tf = tempfile.TemporaryFile()
        d = tf.fileno()
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys; sys.stdout.write("orange")'],
                         stdout=d)
        p.wait()
        os.lseek(d, 0, 0)
        self.assertEqual(os.read(d, 1024), "orange")

    def test_stdout_fileobj(self):
        # stdout is set to open file object
        tf = tempfile.TemporaryFile()
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys; sys.stdout.write("orange")'],
                         stdout=tf)
        p.wait()
        tf.seek(0)
        self.assertEqual(tf.read(), "orange")

    def test_stderr_pipe(self):
        # stderr redirection
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys; sys.stderr.write("strawberry")'],
                         stderr=subprocess.PIPE)
        self.assertStderrEqual(p.stderr.read(), "strawberry")

    def test_stderr_filedes(self):
        # stderr is set to open file descriptor
        tf = tempfile.TemporaryFile()
        d = tf.fileno()
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys; sys.stderr.write("strawberry")'],
                         stderr=d)
        p.wait()
        os.lseek(d, 0, 0)
        self.assertStderrEqual(os.read(d, 1024), "strawberry")

    def test_stderr_fileobj(self):
        # stderr is set to open file object
        tf = tempfile.TemporaryFile()
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys; sys.stderr.write("strawberry")'],
                         stderr=tf)
        p.wait()
        tf.seek(0)
        self.assertStderrEqual(tf.read(), "strawberry")

    def test_stderr_redirect_with_no_stdout_redirect(self):
        # test stderr=STDOUT while stdout=None (not set)

        # - grandchild prints to stderr
        # - child redirects grandchild's stderr to its stdout
        # - the parent should get grandchild's stderr in child's stdout
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys, subprocess32 as subprocess;'
                              'rc = subprocess.call([sys.executable, "-c",'
                              '    "import sys;"'
                              '    "sys.stderr.write(\'42\')"],'
                              '    stderr=subprocess.STDOUT);'
                              'sys.exit(rc)'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        #NOTE: stdout should get stderr from grandchild
        self.assertStderrEqual(stdout, '42')
        self.assertStderrEqual(stderr, '') # should be empty
        self.assertEqual(p.returncode, 0)

    def test_stdout_stderr_pipe(self):
        # capture stdout and stderr to the same pipe
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys;'
                          'sys.stdout.write("apple");'
                          'sys.stdout.flush();'
                          'sys.stderr.write("orange")'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
        self.assertStderrEqual(p.stdout.read(), "appleorange")

    def test_stdout_stderr_file(self):
        # capture stdout and stderr to the same open file
        tf = tempfile.TemporaryFile()
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys;'
                          'sys.stdout.write("apple");'
                          'sys.stdout.flush();'
                          'sys.stderr.write("orange")'],
                         stdout=tf,
                         stderr=tf)
        p.wait()
        tf.seek(0)
        self.assertStderrEqual(tf.read(), "appleorange")

    def test_stdout_filedes_of_stdout(self):
        # stdout is set to 1 (#1531862).
        # To avoid printing the text on stdout, we do something similar to
        # test_stdout_none (see above).  The parent subprocess calls the child
        # subprocess passing stdout=1, and this test uses stdout=PIPE in
        # order to capture and check the output of the parent. See #11963.
        code = ('import sys, subprocess32; '
                'rc = subprocess32.call([sys.executable, "-c", '
                '    "import os, sys; sys.exit(os.write(sys.stdout.fileno(), '
                     '\'test with stdout=1\'))"], stdout=1); '
                'assert rc == 18')
        p = subprocess.Popen([sys.executable, "-c", yenv + code],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(p.stdout.close)
        self.addCleanup(p.stderr.close)
        out, err = p.communicate()
        self.assertEqual(p.returncode, 0, err)
        self.assertEqual(out.rstrip(), 'test with stdout=1')

    def test_stdout_devnull(self):
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'for i in range(10240):'
                              'print("x" * 1024)'],
                              stdout=subprocess.DEVNULL)
        p.wait()
        self.assertEqual(p.stdout, None)

    def test_stderr_devnull(self):
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys\n'
                              'for i in range(10240):'
                              'sys.stderr.write("x" * 1024)'],
                              stderr=subprocess.DEVNULL)
        p.wait()
        self.assertEqual(p.stderr, None)

    def test_stdin_devnull(self):
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys;'
                              'sys.stdin.read(1)'],
                              stdin=subprocess.DEVNULL)
        p.wait()
        self.assertEqual(p.stdin, None)

    def test_env(self):
        newenv = os.environ.copy()
        newenv["FRUIT"] = "orange"
        p = subprocess.Popen([sys.executable, "-c", yenv +
                               'import sys,os;'
                               'sys.stdout.write(os.getenv("FRUIT"))'],
                              stdout=subprocess.PIPE,
                              env=newenv)
        try:
            stdout, stderr = p.communicate()
            self.assertEqual(stdout, "orange")
        finally:
            p.__exit__(None, None, None)

    def test_empty_env(self):
        """test_empty_env() - verify that env={} is as empty as possible."""

        def is_env_var_to_ignore(n):
            """Determine if an environment variable is under our control."""
            # This excludes some __CF_* and VERSIONER_* keys MacOS insists
            # on adding even when the environment in exec is empty.
            # Gentoo sandboxes also force LD_PRELOAD and SANDBOX_* to exist.
            return ('VERSIONER' in n or '__CF' in n or  # MacOS
                    n == 'LD_PRELOAD' or n.startswith('SANDBOX'))  # Gentoo

        p = subprocess.Popen(
                [sys.executable, '-c',
                 'import os; print(list(os.environ.keys()))'],
                stdout=subprocess.PIPE, env={'Y_PYTHON_ENTRY_POINT': ':main'})
        try:
            stdout, stderr = p.communicate()
            child_env_names = eval(stdout.strip())
            self.assertTrue(isinstance(child_env_names, list),
                            msg=repr(child_env_names))
            child_env_names = [k for k in child_env_names
                               if not is_env_var_to_ignore(k)]
            self.assertEqual(child_env_names, [])
        finally:
            p.__exit__(None, None, None)

    def test_communicate_stdin(self):
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys;'
                              'sys.exit(sys.stdin.read() == "pear")'],
                             stdin=subprocess.PIPE)
        p.communicate("pear")
        self.assertEqual(p.returncode, 1)

    def test_communicate_stdout(self):
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys; sys.stdout.write("pineapple")'],
                             stdout=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
        self.assertEqual(stdout, "pineapple")
        self.assertEqual(stderr, None)

    def test_communicate_stderr(self):
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys; sys.stderr.write("pineapple")'],
                             stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
        self.assertEqual(stdout, None)
        self.assertStderrEqual(stderr, "pineapple")

    def test_communicate(self):
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys,os;'
                          'sys.stderr.write("pineapple");'
                          'sys.stdout.write(sys.stdin.read())'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate("banana")
        self.assertEqual(stdout, "banana")
        self.assertStderrEqual(stderr, "pineapple")

    def test_communicate_timeout(self):
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys,os,time;'
                              'sys.stderr.write("pineapple\\n");'
                              'time.sleep(1);'
                              'sys.stderr.write("pear\\n");'
                              'sys.stdout.write(sys.stdin.read())'],
                             universal_newlines=True,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        self.assertRaises(subprocess.TimeoutExpired, p.communicate, u"banana",
                          timeout=0.3)
        # Make sure we can keep waiting for it, and that we get the whole output
        # after it completes.
        (stdout, stderr) = p.communicate()
        self.assertEqual(stdout, "banana")
        self.assertStderrEqual(stderr.encode(), "pineapple\npear\n")

    def test_communicate_timeout_large_ouput(self):
        # Test a expring timeout while the child is outputting lots of data.
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys,os,time;'
                              'sys.stdout.write("a" * (64 * 1024));'
                              'time.sleep(0.2);'
                              'sys.stdout.write("a" * (64 * 1024));'
                              'time.sleep(0.2);'
                              'sys.stdout.write("a" * (64 * 1024));'
                              'time.sleep(0.2);'
                              'sys.stdout.write("a" * (64 * 1024));'],
                             stdout=subprocess.PIPE)
        self.assertRaises(subprocess.TimeoutExpired, p.communicate, timeout=0.4)
        (stdout, _) = p.communicate()
        self.assertEqual(len(stdout), 4 * 64 * 1024)

    # Test for the fd leak reported in http://bugs.python.org/issue2791.
    def test_communicate_pipe_fd_leak(self):
        for stdin_pipe in (False, True):
            for stdout_pipe in (False, True):
                for stderr_pipe in (False, True):
                    options = {}
                    if stdin_pipe:
                        options['stdin'] = subprocess.PIPE
                    if stdout_pipe:
                        options['stdout'] = subprocess.PIPE
                    if stderr_pipe:
                        options['stderr'] = subprocess.PIPE
                    if not options:
                        continue
                    p = subprocess.Popen((sys.executable, "-c", yenv + "pass"), **options)
                    p.communicate()
                    if p.stdin is not None:
                        self.assertTrue(p.stdin.closed)
                    if p.stdout is not None:
                        self.assertTrue(p.stdout.closed)
                    if p.stderr is not None:
                        self.assertTrue(p.stderr.closed)

    def test_communicate_returns(self):
        # communicate() should return None if no redirection is active
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              "import sys; sys.exit(47)"])
        (stdout, stderr) = p.communicate()
        self.assertEqual(stdout, None)
        self.assertEqual(stderr, None)

    def test_communicate_pipe_buf(self):
        # communicate() with writes larger than pipe_buf
        # This test will probably deadlock rather than fail, if
        # communicate() does not work properly.
        x, y = os.pipe()
        if mswindows:
            pipe_buf = 512
        else:
            pipe_buf = os.fpathconf(x, "PC_PIPE_BUF")
        os.close(x)
        os.close(y)
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys,os;'
                          'sys.stdout.write(sys.stdin.read(47));'
                          'sys.stderr.write("xyz"*%d);'
                          'sys.stdout.write(sys.stdin.read())' % pipe_buf],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
        string_to_write = "abc"*pipe_buf
        (stdout, stderr) = p.communicate(string_to_write)
        self.assertEqual(stdout, string_to_write)

    def test_writes_before_communicate(self):
        # stdin.write before communicate()
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys,os;'
                          'sys.stdout.write(sys.stdin.read())'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
        p.stdin.write("banana")
        (stdout, stderr) = p.communicate("split")
        self.assertEqual(stdout, "bananasplit")
        self.assertStderrEqual(stderr, "")

    def test_universal_newlines(self):
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys,os;' + SETBINARY +
                          'sys.stdout.write("line1\\n");'
                          'sys.stdout.flush();'
                          'sys.stdout.write("line2\\r");'
                          'sys.stdout.flush();'
                          'sys.stdout.write("line3\\r\\n");'
                          'sys.stdout.flush();'
                          'sys.stdout.write("line4\\r");'
                          'sys.stdout.flush();'
                          'sys.stdout.write("\\nline5");'
                          'sys.stdout.flush();'
                          'sys.stdout.write("\\nline6");'],
                         stdout=subprocess.PIPE,
                         universal_newlines=1)
        stdout = p.stdout.read()
        if hasattr(file, 'newlines'):
            # Interpreter with universal newline support
            self.assertEqual(stdout,
                             "line1\nline2\nline3\nline4\nline5\nline6")
        else:
            # Interpreter without universal newline support
            self.assertEqual(stdout,
                             "line1\nline2\rline3\r\nline4\r\nline5\nline6")

    def test_universal_newlines_communicate(self):
        # universal newlines through communicate()
        p = subprocess.Popen([sys.executable, "-c", yenv +
                          'import sys,os;' + SETBINARY +
                          'sys.stdout.write("line1\\n");'
                          'sys.stdout.flush();'
                          'sys.stdout.write("line2\\r");'
                          'sys.stdout.flush();'
                          'sys.stdout.write("line3\\r\\n");'
                          'sys.stdout.flush();'
                          'sys.stdout.write("line4\\r");'
                          'sys.stdout.flush();'
                          'sys.stdout.write("\\nline5");'
                          'sys.stdout.flush();'
                          'sys.stdout.write("\\nline6");'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=1)
        (stdout, stderr) = p.communicate()
        if hasattr(file, 'newlines'):
            # Interpreter with universal newline support
            self.assertEqual(stdout,
                             "line1\nline2\nline3\nline4\nline5\nline6")
        else:
            # Interpreter without universal newline support
            self.assertEqual(stdout,
                             "line1\nline2\rline3\r\nline4\r\nline5\nline6")

    def test_universal_newlines_communicate_input_none(self):
        # Test communicate(input=None) with universal newlines.
        #
        # We set stdout to PIPE because, as of this writing, a different
        # code path is tested when the number of pipes is zero or one.
        p = subprocess.Popen([sys.executable, "-c", yenv + "pass"],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
        p.communicate()
        self.assertEqual(p.returncode, 0)

    def test_no_leaking(self):
        # Make sure we leak no resources
        if not hasattr(test_support, "is_resource_enabled") \
               or test_support.is_resource_enabled("subprocess") and not mswindows:
            max_handles = 1026 # too much for most UNIX systems
        else:
            max_handles = 65
        for i in range(max_handles):
            p = subprocess.Popen([sys.executable, "-c", yenv +
                    "import sys;sys.stdout.write(sys.stdin.read())"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
            data = p.communicate("lime")[0]
            self.assertEqual(data, "lime")

    def test_universal_newlines_communicate_stdin_stdout_stderr(self):
        # universal newlines through communicate(), with stdin, stdout, stderr
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys,os;' + SETBINARY + '''\nif True:
                                  s = sys.stdin.readline()
                                  sys.stdout.write(s)
                                  sys.stdout.write("line2\\r")
                                  sys.stderr.write("eline2\\n")
                                  s = sys.stdin.read()
                                  sys.stdout.write(s)
                                  sys.stdout.write("line4\\n")
                                  sys.stdout.write("line5\\r\\n")
                                  sys.stderr.write("eline6\\r")
                                  sys.stderr.write("eline7\\r\\nz")
                              '''],
                             stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
        self.addCleanup(p.stdout.close)
        self.addCleanup(p.stderr.close)
        (stdout, stderr) = p.communicate(u"line1\nline3\n")
        self.assertEqual(p.returncode, 0)
        self.assertEqual(u"line1\nline2\nline3\nline4\nline5\n", stdout)
        # Python debug build push something like "[42442 refs]\n"
        # to stderr at exit of subprocess.
        # Don't use assertStderrEqual because it strips CR and LF from output.
        self.assertTrue(stderr.startswith(u"eline2\neline6\neline7\n"))

    def test_list2cmdline(self):
        self.assertEqual(subprocess.list2cmdline(['a b c', 'd', 'e']),
                         '"a b c" d e')
        self.assertEqual(subprocess.list2cmdline(['ab"c', '\\', 'd']),
                         'ab\\"c \\ d')
        self.assertEqual(subprocess.list2cmdline(['ab"c', ' \\', 'd']),
                         'ab\\"c " \\\\" d')
        self.assertEqual(subprocess.list2cmdline(['a\\\\\\b', 'de fg', 'h']),
                         'a\\\\\\b "de fg" h')
        self.assertEqual(subprocess.list2cmdline(['a\\"b', 'c', 'd']),
                         'a\\\\\\"b c d')
        self.assertEqual(subprocess.list2cmdline(['a\\\\b c', 'd', 'e']),
                         '"a\\\\b c" d e')
        self.assertEqual(subprocess.list2cmdline(['a\\\\b\\ c', 'd', 'e']),
                         '"a\\\\b\\ c" d e')
        self.assertEqual(subprocess.list2cmdline(['ab', '']),
                         'ab ""')


    def test_poll(self):
        p = subprocess.Popen([sys.executable,
                          "-c", yenv + "import time; time.sleep(1)"])
        count = 0
        while p.poll() is None:
            time.sleep(0.1)
            count += 1
        # We expect that the poll loop probably went around about 10 times,
        # but, based on system scheduling we can't control, it's possible
        # poll() never returned None.  It "should be" very rare that it
        # didn't go around at least twice.
        self.assert_(count >= 2)
        # Subsequent invocations should just return the returncode
        self.assertEqual(p.poll(), 0)


    def test_wait(self):
        p = subprocess.Popen([sys.executable,
                          "-c", yenv + "import time; time.sleep(2)"])
        self.assertEqual(p.wait(), 0)
        # Subsequent invocations should just return the returncode
        self.assertEqual(p.wait(), 0)


    def test_wait_timeout(self):
        p = subprocess.Popen([sys.executable,
                              "-c", yenv + "import time; time.sleep(0.1)"])
        try:
            p.wait(timeout=0.01)
        except subprocess.TimeoutExpired, e:
            self.assertIn("0.01", str(e))  # For coverage of __str__.
        else:
            self.fail("subprocess.TimeoutExpired expected but not raised.")
        self.assertEqual(p.wait(timeout=2), 0)


    def test_invalid_bufsize(self):
        # an invalid type of the bufsize argument should raise
        # TypeError.
        try:
            subprocess.Popen([sys.executable, "-c", yenv + "pass"], "orange")
        except TypeError:
            pass

    def test_leaking_fds_on_error(self):
        # see bug #5179: Popen leaks file descriptors to PIPEs if
        # the child fails to execute; this will eventually exhaust
        # the maximum number of open fds. 1024 seems a very common
        # value for that limit, but Windows has 2048, so we loop
        # 1024 times (each call leaked two fds).
        for i in range(1024):
            # Windows raises IOError.  Others raise OSError.
            try:
                subprocess.Popen(['nonexisting_i_hope'],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            except EnvironmentError, c:
                if c.errno != 2:  # ignore "no such file"
                    raise

    #@unittest.skipIf(threading is None, "threading required")
    def test_threadsafe_wait(self):
        """Issue21291: Popen.wait() needs to be threadsafe for returncode."""
        proc = subprocess.Popen([sys.executable, '-c', yenv +
                                 'import time; time.sleep(12)'])
        self.assertEqual(proc.returncode, None)
        results = []

        def kill_proc_timer_thread():
            results.append(('thread-start-poll-result', proc.poll()))
            # terminate it from the thread and wait for the result.
            proc.kill()
            proc.wait()
            results.append(('thread-after-kill-and-wait', proc.returncode))
            # this wait should be a no-op given the above.
            proc.wait()
            results.append(('thread-after-second-wait', proc.returncode))

        # This is a timing sensitive test, the failure mode is
        # triggered when both the main thread and this thread are in
        # the wait() call at once.  The delay here is to allow the
        # main thread to most likely be blocked in its wait() call.
        t = threading.Timer(0.2, kill_proc_timer_thread)
        t.start()

        if mswindows:
            expected_errorcode = 1
        else:
            # Should be -9 because of the proc.kill() from the thread.
            expected_errorcode = -9

        # Wait for the process to finish; the thread should kill it
        # long before it finishes on its own.  Supplying a timeout
        # triggers a different code path for better coverage.
        proc.wait(timeout=20)
        self.assertEqual(proc.returncode, expected_errorcode,
                         msg="unexpected result in wait from main thread")

        # This should be a no-op with no change in returncode.
        proc.wait()
        self.assertEqual(proc.returncode, expected_errorcode,
                         msg="unexpected result in second main wait.")

        t.join()
        # Ensure that all of the thread results are as expected.
        # When a race condition occurs in wait(), the returncode could
        # be set by the wrong thread that doesn't actually have it
        # leading to an incorrect value.
        self.assertEqual([('thread-start-poll-result', None),
                          ('thread-after-kill-and-wait', expected_errorcode),
                          ('thread-after-second-wait', expected_errorcode)],
                         results)

    def test_issue8780(self):
        # Ensure that stdout is inherited from the parent
        # if stdout=PIPE is not used
        code = ';'.join((
            'import subprocess32, sys',
            'retcode = subprocess32.call('
                "[sys.executable, '-c', 'print(\"Hello World!\")'])",
            'assert retcode == 0'))
        output = subprocess.check_output([sys.executable, '-c', yenv + code])
        self.assert_(output.startswith('Hello World!'), output)

    def test_communicate_epipe(self):
        # Issue 10963: communicate() should hide EPIPE
        p = subprocess.Popen([sys.executable, "-c", yenv + 'pass'],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        self.addCleanup(p.stdout.close)
        self.addCleanup(p.stderr.close)
        self.addCleanup(p.stdin.close)
        p.communicate(b"x" * 2**20)

    def test_communicate_epipe_only_stdin(self):
        # Issue 10963: communicate() should hide EPIPE
        p = subprocess.Popen([sys.executable, "-c", yenv + 'pass'],
                             stdin=subprocess.PIPE)
        self.addCleanup(p.stdin.close)
        p.wait()
        p.communicate(b"x" * 2**20)

    if not mswindows:  # Signal tests are POSIX specific.
        def test_communicate_eintr(self):
            # Issue #12493: communicate() should handle EINTR
            def handler(signum, frame):
                pass
            old_handler = signal.signal(signal.SIGALRM, handler)
            self.addCleanup(signal.signal, signal.SIGALRM, old_handler)

            # the process is running for 2 seconds
            args = [sys.executable, "-c", yenv + 'import time; time.sleep(2)']
            for stream in ('stdout', 'stderr'):
                kw = {stream: subprocess.PIPE}
                process = subprocess.Popen(args, **kw)
                try:
                    signal.alarm(1)
                    # communicate() will be interrupted by SIGALRM
                    process.communicate()
                finally:
                    process.__exit__(None, None, None)


    # This test is Linux-ish specific for simplicity to at least have
    # some coverage.  It is not a platform specific bug.
    #@unittest.skipUnless(os.path.isdir('/proc/%d/fd' % os.getpid()),
    #                     "Linux specific")
    def test_failed_child_execute_fd_leak(self):
        """Test for the fork() failure fd leak reported in issue16327."""
        if not os.path.isdir('/proc/%d/fd' % os.getpid()):
            self.skipTest("Linux specific")
        fd_directory = '/proc/%d/fd' % os.getpid()
        fds_before_popen = os.listdir(fd_directory)
        try:
            PopenExecuteChildRaises(
                    [sys.executable, '-c', yenv + 'pass'], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except PopenTestException:
            pass  # Yay!  Because 2.4 doesn't support with statements.
        else:
            self.fail("PopenTestException expected but not raised.")

        # NOTE: This test doesn't verify that the real _execute_child
        # does not close the file descriptors itself on the way out
        # during an exception.  Code inspection has confirmed that.

        fds_after_exception = os.listdir(fd_directory)
        self.assertEqual(fds_before_popen, fds_after_exception)


class RunFuncTestCase(BaseTestCase):
    def run_python(self, code, **kwargs):
        """Run Python code in a subprocess using subprocess.run"""
        argv = [sys.executable, "-c", yenv + code]
        return subprocess.run(argv, **kwargs)

    def test_returncode(self):
        # call() function with sequence argument
        cp = self.run_python("import sys; sys.exit(47)")
        self.assertEqual(cp.returncode, 47)
        try:
            cp.check_returncode()
        except subprocess.CalledProcessError:
            pass
        else:
            self.fail("CalledProcessError not raised")

    def test_check(self):
        try:
            self.run_python("import sys; sys.exit(47)", check=True)
        except subprocess.CalledProcessError, exception:
            self.assertEqual(exception.returncode, 47)
        else:
            self.fail("CalledProcessError not raised")

    def test_check_zero(self):
        # check_returncode shouldn't raise when returncode is zero
        cp = self.run_python("import sys; sys.exit(0)", check=True)
        self.assertEqual(cp.returncode, 0)

    def test_timeout(self):
        # run() function with timeout argument; we want to test that the child
        # process gets killed when the timeout expires.  If the child isn't
        # killed, this call will deadlock since subprocess.run waits for the
        # child.
        try:
            self.run_python("while True: pass", timeout=0.0001)
        except subprocess.TimeoutExpired:
            pass
        else:
            self.fail("TimeoutExpired not raised")

    def test_capture_stdout(self):
        # capture stdout with zero return code
        cp = self.run_python("print('BDFL')", stdout=subprocess.PIPE)
        self.assertIn('BDFL', cp.stdout)

    def test_capture_stderr(self):
        cp = self.run_python("import sys; sys.stderr.write('BDFL')",
                             stderr=subprocess.PIPE)
        self.assertIn('BDFL', cp.stderr)

    def test_check_output_stdin_arg(self):
        # run() can be called with stdin set to a file
        tf = tempfile.TemporaryFile()
        self.addCleanup(tf.close)
        tf.write('pear')
        tf.seek(0)
        cp = self.run_python(
                 "import sys; sys.stdout.write(sys.stdin.read().upper())",
                stdin=tf, stdout=subprocess.PIPE)
        self.assertIn('PEAR', cp.stdout)

    def test_check_output_input_arg(self):
        # check_output() can be called with input set to a string
        cp = self.run_python(
                "import sys; sys.stdout.write(sys.stdin.read().upper())",
                input='pear', stdout=subprocess.PIPE)
        self.assertIn('PEAR', cp.stdout)

    def test_check_output_stdin_with_input_arg(self):
        # run() refuses to accept 'stdin' with 'input'
        tf = tempfile.TemporaryFile()
        self.addCleanup(tf.close)
        tf.write('pear')
        tf.seek(0)
        try:
            output = self.run_python("print('will not be run')",
                                     stdin=tf, input='hare')
        except ValueError, exception:
            self.assertIn('stdin', exception.args[0])
            self.assertIn('input', exception.args[0])
        else:
            self.fail("Expected ValueError when stdin and input args supplied.")

    def test_check_output_timeout(self):
        try:
            cp = self.run_python((
                     "import sys, time\n"
                     "sys.stdout.write('BDFL')\n"
                     "sys.stdout.flush()\n"
                     "time.sleep(3600)"),
                    # Some heavily loaded buildbots (sparc Debian 3.x) require
                    # this much time to start and print.
                    timeout=3, stdout=subprocess.PIPE)
        except subprocess.TimeoutExpired, exception:
            self.assertEqual(exception.output, 'BDFL')
            # output is aliased to stdout
            self.assertEqual(exception.stdout, 'BDFL')
        else:
            self.fail("TimeoutExpired not raised")

    def test_run_kwargs(self):
        newenv = os.environ.copy()
        newenv["FRUIT"] = "banana"
        cp = self.run_python(('import sys, os;'
                              'os.getenv("FRUIT")=="banana" and sys.exit(33) or sys.exit(31)'),
                             env=newenv)
        self.assertEqual(cp.returncode, 33)


# context manager
class _SuppressCoreFiles(object):
    """Try to prevent core files from being created."""
    old_limit = None

    def __enter__(self):
        """Try to save previous ulimit, then set it to (0, 0)."""
        try:
            import resource
            self.old_limit = resource.getrlimit(resource.RLIMIT_CORE)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ImportError, ValueError, resource.error):
            pass

    def __exit__(self, *args):
        """Return core file behavior to default."""
        if self.old_limit is None:
            return
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, self.old_limit)
        except (ImportError, ValueError, resource.error):
            pass


#@unittest.skipIf(mswindows, "POSIX specific tests")
class POSIXProcessTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self._nonexistent_dir = "/_this/pa.th/does/not/exist"

    def _get_chdir_exception(self):
        try:
            os.chdir(self._nonexistent_dir)
        except OSError, e:
            # This avoids hard coding the errno value or the OS perror()
            # string and instead capture the exception that we want to see
            # below for comparison.
            desired_exception = e
            desired_exception.strerror += ': ' + repr(self._nonexistent_dir)
        else:
            self.fail("chdir to nonexistant directory %s succeeded." %
                      self._nonexistent_dir)
        return desired_exception

    def test_exception_cwd(self):
        """Test error in the child raised in the parent for a bad cwd."""
        desired_exception = self._get_chdir_exception()
        try:
            p = subprocess.Popen([sys.executable, "-c", yenv + ""],
                                 cwd=self._nonexistent_dir)
        except OSError, e:
            # Test that the child process chdir failure actually makes
            # it up to the parent process as the correct exception.
            self.assertEqual(desired_exception.errno, e.errno)
            self.assertEqual(desired_exception.strerror, e.strerror)
        else:
            self.fail("Expected OSError: %s" % desired_exception)

    def test_exception_bad_executable(self):
        """Test error in the child raised in the parent for a bad executable."""
        desired_exception = self._get_chdir_exception()
        try:
            p = subprocess.Popen([sys.executable, "-c", yenv + ""],
                                 executable=self._nonexistent_dir)
        except OSError, e:
            # Test that the child process exec failure actually makes
            # it up to the parent process as the correct exception.
            self.assertEqual(desired_exception.errno, e.errno)
            self.assertEqual(desired_exception.strerror, e.strerror)
        else:
            self.fail("Expected OSError: %s" % desired_exception)

    def test_exception_bad_args_0(self):
        """Test error in the child raised in the parent for a bad args[0]."""
        desired_exception = self._get_chdir_exception()
        try:
            p = subprocess.Popen([self._nonexistent_dir, "-c", yenv + ""])
        except OSError, e:
            # Test that the child process exec failure actually makes
            # it up to the parent process as the correct exception.
            self.assertEqual(desired_exception.errno, e.errno)
            self.assertEqual(desired_exception.strerror, e.strerror)
        else:
            self.fail("Expected OSError: %s" % desired_exception)

    #@unittest.skipIf(not os.path.exists('/proc/self/status'))
    def test_restore_signals(self):
        if not os.path.exists('/proc/self/status'):
            print("SKIP - Functional test requires /proc/self/status.")
            return
        # Blindly assume that cat exists on systems with /proc/self/status...
        default_proc_status = subprocess.check_output(
                ['cat', '/proc/self/status'],
                restore_signals=False)
        for line in default_proc_status.splitlines():
            if line.startswith(b'SigIgn'):
                default_sig_ign_mask = line
                break
        else:
            self.skipTest("SigIgn not found in /proc/self/status.")
        restored_proc_status = subprocess.check_output(
                ['cat', '/proc/self/status'],
                restore_signals=True)
        for line in restored_proc_status.splitlines():
            if line.startswith(b'SigIgn'):
                restored_sig_ign_mask = line
                break
        # restore_signals=True should've unblocked SIGPIPE and friends.
        self.assertNotEqual(default_sig_ign_mask, restored_sig_ign_mask)

    def test_start_new_session(self):
        # For code coverage of calling setsid().  We don't care if we get an
        # EPERM error from it depending on the test execution environment, that
        # still indicates that it was called.
        try:
            output = subprocess.check_output(
                    [sys.executable, "-c", yenv +
                     "import os; print(os.getpgid(os.getpid()))"],
                    start_new_session=True)
        except OSError, e:
            if e.errno != errno.EPERM:
                raise
        else:
            parent_pgid = os.getpgid(os.getpid())
            child_pgid = int(output)
            self.assertNotEqual(parent_pgid, child_pgid)

    def test_run_abort(self):
        # returncode handles signal termination
        scf = _SuppressCoreFiles()
        scf.__enter__()
        try:
            p = subprocess.Popen([sys.executable, "-c", yenv +
                                  "import os; os.abort()"])
            p.wait()
        finally:
            scf.__exit__()
        self.assertEqual(-p.returncode, signal.SIGABRT)

    def test_preexec(self):
        # DISCLAIMER: Setting environment variables is *not* a good use
        # of a preexec_fn.  This is merely a test.
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              "import sys, os;"
                              "sys.stdout.write(os.getenv('FRUIT'))"],
                             stdout=subprocess.PIPE,
                             preexec_fn=lambda: os.putenv("FRUIT", "apple"))
        self.assertEqual(p.stdout.read(), "apple")

    def test_preexec_exception(self):
        def raise_it():
            raise ValueError("What if two swallows carried a coconut?")
        try:
            p = subprocess.Popen([sys.executable, "-c", yenv + ""],
                                 preexec_fn=raise_it)
        except RuntimeError, e:
            self.assertTrue(
                    subprocess._posixsubprocess,
                    "Expected a ValueError from the preexec_fn")
        except ValueError, e:
            self.assertIn("coconut", e.args[0])
        else:
            self.fail("Exception raised by preexec_fn did not make it "
                      "to the parent process.")

    class _TestExecuteChildPopen(subprocess.Popen):
        """Used to test behavior at the end of _execute_child."""
        def __init__(self, testcase, *args, **kwargs):
            self._testcase = testcase
            subprocess.Popen.__init__(self, *args, **kwargs)

        def _execute_child(self, *args, **kwargs):
            try:
                subprocess.Popen._execute_child(self, *args, **kwargs)
            finally:
                # Open a bunch of file descriptors and verify that
                # none of them are the same as the ones the Popen
                # instance is using for stdin/stdout/stderr.
                devzero_fds = [os.open("/dev/zero", os.O_RDONLY)
                               for _ in range(8)]
                try:
                    for fd in devzero_fds:
                        self._testcase.assertNotIn(
                                fd, (self.stdin.fileno(), self.stdout.fileno(),
                                     self.stderr.fileno()),
                                msg="At least one fd was closed early.")
                finally:
                    map(os.close, devzero_fds)

    #@unittest.skipIf(not os.path.exists("/dev/zero"), "/dev/zero required.")
    def test_preexec_errpipe_does_not_double_close_pipes(self):
        """Issue16140: Don't double close pipes on preexec error."""

        def raise_it():
            raise RuntimeError("force the _execute_child() errpipe_data path.")

        try:
            self._TestExecuteChildPopen(
                        self, [sys.executable, "-c", yenv + "pass"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, preexec_fn=raise_it)
        except RuntimeError:
            pass  # Yay!  Because 2.4 doesn't support with statements.
        else:
            self.fail("RuntimeError expected but not raised.")

    #@unittest.skipUnless(gc, "Requires a gc module.")
    def test_preexec_gc_module_failure(self):
        # This tests the code that disables garbage collection if the child
        # process will execute any Python.
        def raise_runtime_error():
            raise RuntimeError("this shouldn't escape")
        enabled = gc.isenabled()
        orig_gc_disable = gc.disable
        orig_gc_isenabled = gc.isenabled
        try:
            gc.disable()
            self.assertFalse(gc.isenabled())
            subprocess.call([sys.executable, '-c', yenv + ''],
                            preexec_fn=lambda: None)
            self.assertFalse(gc.isenabled(),
                             "Popen enabled gc when it shouldn't.")

            gc.enable()
            self.assertTrue(gc.isenabled())
            subprocess.call([sys.executable, '-c', yenv + ''],
                            preexec_fn=lambda: None)
            self.assertTrue(gc.isenabled(), "Popen left gc disabled.")

            gc.disable = raise_runtime_error
            self.assertRaises(RuntimeError, subprocess.Popen,
                              [sys.executable, '-c', yenv + ''],
                              preexec_fn=lambda: None)

            del gc.isenabled  # force an AttributeError
            self.assertRaises(AttributeError, subprocess.Popen,
                              [sys.executable, '-c', yenv + ''],
                              preexec_fn=lambda: None)
        finally:
            gc.disable = orig_gc_disable
            gc.isenabled = orig_gc_isenabled
            if not enabled:
                gc.disable()

    def test_args_string(self):
        # args is a string
        f, fname = mkstemp()
        os.write(f, "#!/bin/sh\n")
        os.write(f, "exec '%s' -c 'import sys; sys.exit(47)'\n" %
                    sys.executable)
        os.close(f)
        os.chmod(fname, 0700)
        p = subprocess.Popen(fname)
        p.wait()
        os.remove(fname)
        self.assertEqual(p.returncode, 47)

    def test_invalid_args(self):
        # invalid arguments should raise ValueError
        self.assertRaises(ValueError, subprocess.call,
                          [sys.executable, "-c", yenv +
                           "import sys; sys.exit(47)"],
                          startupinfo=47)
        self.assertRaises(ValueError, subprocess.call,
                          [sys.executable, "-c", yenv +
                           "import sys; sys.exit(47)"],
                          creationflags=47)

    def test_shell_sequence(self):
        # Run command through the shell (sequence)
        newenv = os.environ.copy()
        newenv["FRUIT"] = "apple"
        p = subprocess.Popen(["echo $FRUIT"], shell=1,
                             stdout=subprocess.PIPE,
                             env=newenv)
        self.assertEqual(p.stdout.read().strip(), "apple")

    def test_shell_string(self):
        # Run command through the shell (string)
        newenv = os.environ.copy()
        newenv["FRUIT"] = "apple"
        p = subprocess.Popen("echo $FRUIT", shell=1,
                             stdout=subprocess.PIPE,
                             env=newenv)
        self.assertEqual(p.stdout.read().strip(), "apple")

    def test_call_string(self):
        # call() function with string argument on UNIX
        f, fname = mkstemp()
        os.write(f, "#!/bin/sh\n")
        os.write(f, "exec '%s' -c 'import sys; sys.exit(47)'\n" %
                    sys.executable)
        os.close(f)
        os.chmod(fname, 0700)
        rc = subprocess.call(fname)
        os.remove(fname)
        self.assertEqual(rc, 47)

    def test_specific_shell(self):
        # Issue #9265: Incorrect name passed as arg[0].
        shells = []
        for prefix in ['/bin', '/usr/bin/', '/usr/local/bin']:
            for name in ['bash', 'ksh']:
                sh = os.path.join(prefix, name)
                if os.path.isfile(sh):
                    shells.append(sh)
        if not shells: # Will probably work for any shell but csh.
            self.skipTest("bash or ksh required for this test")
        sh = '/bin/sh'
        if os.path.isfile(sh) and not os.path.islink(sh):
            # Test will fail if /bin/sh is a symlink to csh.
            shells.append(sh)
        for sh in shells:
            p = subprocess.Popen("echo $0", executable=sh, shell=True,
                                 stdout=subprocess.PIPE)
            self.assertEqual(p.stdout.read().strip(), sh)

    def _kill_process(self, method, *args):
        # Do not inherit file handles from the parent.
        # It should fix failures on some platforms.
        p = subprocess.Popen([sys.executable, "-c", yenv + """if 1:
                             import sys, time
                             sys.stdout.write('x\\n')
                             sys.stdout.flush()
                             time.sleep(30)
                             """],
                             close_fds=True,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        # Wait for the interpreter to be completely initialized before
        # sending any signal.
        p.stdout.read(1)
        getattr(p, method)(*args)
        return p

    def test_send_signal(self):
        p = self._kill_process('send_signal', signal.SIGINT)
        _, stderr = p.communicate()
        self.assertIn('KeyboardInterrupt', stderr)
        self.assertNotEqual(p.wait(), 0)

    def test_kill(self):
        p = self._kill_process('kill')
        _, stderr = p.communicate()
        self.assertStderrEqual(stderr, '')
        self.assertEqual(p.wait(), -signal.SIGKILL)

    def test_terminate(self):
        p = self._kill_process('terminate')
        _, stderr = p.communicate()
        self.assertStderrEqual(stderr, '')
        self.assertEqual(p.wait(), -signal.SIGTERM)

    def check_close_std_fds(self, fds):
        # Issue #9905: test that subprocess pipes still work properly with
        # some standard fds closed
        stdin = 0
        newfds = []
        for a in fds:
            b = os.dup(a)
            newfds.append(b)
            if a == 0:
                stdin = b
        try:
            for fd in fds:
                os.close(fd)
            out, err = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys;'
                              'sys.stdout.write("apple");'
                              'sys.stdout.flush();'
                              'sys.stderr.write("orange")'],
                       stdin=stdin,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE).communicate()
            err = strip_python_stderr(err)
            self.assertEqual((out, err), ('apple', 'orange'))
        finally:
            for b, a in zip(newfds, fds):
                os.dup2(b, a)
            for b in newfds:
                os.close(b)

    def test_close_fd_0(self):
        self.check_close_std_fds([0])

    def test_close_fd_1(self):
        self.check_close_std_fds([1])

    def test_close_fd_2(self):
        self.check_close_std_fds([2])

    def test_close_fds_0_1(self):
        self.check_close_std_fds([0, 1])

    def test_close_fds_0_2(self):
        self.check_close_std_fds([0, 2])

    def test_close_fds_1_2(self):
        self.check_close_std_fds([1, 2])

    def test_close_fds_0_1_2(self):
        # Issue #10806: test that subprocess pipes still work properly with
        # all standard fds closed.
        self.check_close_std_fds([0, 1, 2])

    def check_swap_fds(self, stdin_no, stdout_no, stderr_no):
        # open up some temporary files
        temps = [mkstemp() for i in range(3)]
        temp_fds = [fd for fd, fname in temps]
        try:
            # unlink the files -- we won't need to reopen them
            for fd, fname in temps:
                os.unlink(fname)

            # save a copy of the standard file descriptors
            saved_fds = [os.dup(fd) for fd in range(3)]
            try:
                # duplicate the temp files over the standard fd's 0, 1, 2
                for fd, temp_fd in enumerate(temp_fds):
                    os.dup2(temp_fd, fd)

                # write some data to what will become stdin, and rewind
                os.write(stdin_no, "STDIN")
                os.lseek(stdin_no, 0, 0)

                # now use those files in the given order, so that subprocess
                # has to rearrange them in the child
                p = subprocess.Popen([sys.executable, "-c", yenv +
                    'import sys; got = sys.stdin.read();'
                    'sys.stdout.write("got %s"%got); sys.stderr.write("err")'],
                    stdin=stdin_no,
                    stdout=stdout_no,
                    stderr=stderr_no)
                p.wait()

                for fd in temp_fds:
                    os.lseek(fd, 0, 0)

                out = os.read(stdout_no, 1024)
                err = os.read(stderr_no, 1024)
            finally:
                for std, saved in enumerate(saved_fds):
                    os.dup2(saved, std)
                    os.close(saved)

            self.assertEqual(out, "got STDIN")
            self.assertStderrEqual(err, "err")

        finally:
            for fd in temp_fds:
                os.close(fd)

    # When duping fds, if there arises a situation where one of the fds is
    # either 0, 1 or 2, it is possible that it is overwritten (#12607).
    # This tests all combinations of this.
    def test_swap_fds(self):
        self.check_swap_fds(0, 1, 2)
        self.check_swap_fds(0, 2, 1)
        self.check_swap_fds(1, 0, 2)
        self.check_swap_fds(1, 2, 0)
        self.check_swap_fds(2, 0, 1)
        self.check_swap_fds(2, 1, 0)

    def test_small_errpipe_write_fd(self):
        """Issue #15798: Popen should work when stdio fds are available."""
        new_stdin = os.dup(0)
        new_stdout = os.dup(1)
        try:
            os.close(0)
            os.close(1)

            subprocess.Popen([
                    sys.executable, "-c", yenv + "pass"]).wait()
        finally:
            # Restore original stdin and stdout
            os.dup2(new_stdin, 0)
            os.dup2(new_stdout, 1)
            os.close(new_stdin)
            os.close(new_stdout)

    def test_remapping_std_fds(self):
        # open up some temporary files
        temps = [mkstemp() for i in range(3)]
        try:
            temp_fds = [fd for fd, fname in temps]

            # unlink the files -- we won't need to reopen them
            for fd, fname in temps:
                os.unlink(fname)

            # write some data to what will become stdin, and rewind
            os.write(temp_fds[1], "STDIN")
            os.lseek(temp_fds[1], 0, 0)

            # move the standard file descriptors out of the way
            saved_fds = [os.dup(fd) for fd in range(3)]
            try:
                # duplicate the file objects over the standard fd's
                for fd, temp_fd in enumerate(temp_fds):
                    os.dup2(temp_fd, fd)

                # now use those files in the "wrong" order, so that subprocess
                # has to rearrange them in the child
                p = subprocess.Popen([sys.executable, "-c", yenv +
                    'import sys; got = sys.stdin.read();'
                    'sys.stdout.write("got %s"%got); sys.stderr.write("err")'],
                    stdin=temp_fds[1],
                    stdout=temp_fds[2],
                    stderr=temp_fds[0])
                p.wait()
            finally:
                # restore the original fd's underneath sys.stdin, etc.
                for std, saved in enumerate(saved_fds):
                    os.dup2(saved, std)
                    os.close(saved)

            for fd in temp_fds:
                os.lseek(fd, 0, 0)

            out = os.read(temp_fds[2], 1024)
            err = os.read(temp_fds[0], 1024)
            self.assertEqual(out, "got STDIN")
            self.assertStderrEqual(err, "err")

        finally:
            for fd in temp_fds:
                os.close(fd)

    # NOTE: test_surrogates_error_message makes no sense on python 2.x. omitted.
    # NOTE: test_undecodable_env makes no sense on python 2.x. omitted.
    # NOTE: test_bytes_program makes no sense on python 2.x. omitted.

    if sys.version_info[:2] >= (2,7):
        # Disabling this test on 2.6 and earlier as it fails on Travis CI regardless
        # of LANG=C being set and is not worth the time to figure out why in such a
        # legacy environment..
        #  https://travis-ci.org/google/python-subprocess32/jobs/290065729
        def test_fs_encode_unicode_error(self):
            fs_encoding = sys.getfilesystemencoding()
            if fs_encoding.upper() not in ("ANSI_X3.4-1968", "ASCII"):
                self.skipTest(
                        "Requires a restictive sys.filesystemencoding(), "
                        "not %s.  Run python with LANG=C" % fs_encoding)
            highbit_executable_name = os.path.join(
                    test_support.findfile("testdata"), u"Does\\Not\uDCff\\Exist")
            try:
                subprocess.call([highbit_executable_name])
            except UnicodeEncodeError:
                return
            except RuntimeError, e:
                # The ProcessTestCasePOSIXPurePython version ends up here.  It
                # can't re-construct the unicode error from the child because it
                # doesn't have all the arguments.  BFD.  One doesn't use
                # subprocess32 for the old pure python implementation...
                if "UnicodeEncodeError" not in str(e):
                    self.fail("Expected a RuntimeError whining about how a "
                              "UnicodeEncodeError from the child could not "
                              "be reraised.  Not: %s" % e)
                return
            self.fail("Expected a UnicodeEncodeError to be raised.")

    def test_pipe_cloexec(self):
        sleeper = test_support.findfile("testdata/input_reader.py")
        fd_status = test_support.findfile("testdata/fd_status.py")

        p1 = subprocess.Popen([sys.executable, sleeper],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, close_fds=False)

        self.addCleanup(p1.communicate, '')

        p2 = subprocess.Popen([sys.executable, fd_status],
                              stdout=subprocess.PIPE, close_fds=False)

        output, error = p2.communicate()
        result_fds = set(map(int, output.split(',')))
        unwanted_fds = set([p1.stdin.fileno(), p1.stdout.fileno(),
                            p1.stderr.fileno()])

        self.assertFalse(result_fds & unwanted_fds,
                         "Expected no fds from %r to be open in child, "
                         "found %r" %
                              (unwanted_fds, result_fds & unwanted_fds))

    def test_pipe_cloexec_real_tools(self):
        qcat = test_support.findfile("testdata/qcat.py")
        qgrep = test_support.findfile("testdata/qgrep.py")

        subdata = 'zxcvbn'
        data = subdata * 4 + '\n'

        p1 = subprocess.Popen([sys.executable, qcat],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              close_fds=False)

        p2 = subprocess.Popen([sys.executable, qgrep, subdata],
                              stdin=p1.stdout, stdout=subprocess.PIPE,
                              close_fds=False)

        self.addCleanup(p1.wait)
        self.addCleanup(p2.wait)
        def kill_p1():
            try:
                p1.terminate()
            except ProcessLookupError:
                pass
        def kill_p2():
            try:
                p2.terminate()
            except ProcessLookupError:
                pass
        self.addCleanup(kill_p1)
        self.addCleanup(kill_p2)

        p1.stdin.write(data)
        p1.stdin.close()

        readfiles, ignored1, ignored2 = select.select([p2.stdout], [], [], 10)

        self.assertTrue(readfiles, "The child hung")
        self.assertEqual(p2.stdout.read(), data)

        p1.stdout.close()
        p2.stdout.close()

    def test_close_fds(self):
        fd_status = test_support.findfile("testdata/fd_status.py")

        fds = os.pipe()
        self.addCleanup(os.close, fds[0])
        self.addCleanup(os.close, fds[1])

        open_fds = set(fds)
        # add a bunch more fds
        for _ in range(9):
            fd = os.open("/dev/null", os.O_RDONLY)
            self.addCleanup(os.close, fd)
            open_fds.add(fd)

        p = subprocess.Popen([sys.executable, fd_status],
                             stdout=subprocess.PIPE, close_fds=False)
        output, ignored = p.communicate()
        remaining_fds = set(map(int, output.split(',')))

        self.assertEqual(remaining_fds & open_fds, open_fds,
                         "Some fds were closed")

        p = subprocess.Popen([sys.executable, fd_status],
                             stdout=subprocess.PIPE, close_fds=True)
        output, ignored = p.communicate()
        remaining_fds = set(map(int, output.split(',')))

        self.assertFalse(remaining_fds & open_fds,
                         "Some fds were left open")
        self.assertIn(1, remaining_fds, "Subprocess failed")

        # Keep some of the fd's we opened open in the subprocess.
        # This tests _posixsubprocess.c's proper handling of fds_to_keep.
        fds_to_keep = set(open_fds.pop() for _ in range(8))
        p = subprocess.Popen([sys.executable, fd_status],
                             stdout=subprocess.PIPE, close_fds=True,
                             pass_fds=())
        output, ignored = p.communicate()
        remaining_fds = set(map(int, output.split(',')))

        self.assertFalse(remaining_fds & fds_to_keep & open_fds,
                         "Some fds not in pass_fds were left open")
        self.assertIn(1, remaining_fds, "Subprocess failed")


    def test_close_fds_when_max_fd_is_lowered(self):
        """Confirm that issue21618 is fixed (may fail under valgrind)."""
        fd_status = test_support.findfile("testdata/fd_status.py")

        open_fds = set()
        # Add a bunch more fds to pass down.
        for _ in range(40):
            fd = os.open("/dev/null", os.O_RDONLY)
            open_fds.add(fd)

        # Leave a two pairs of low ones available for use by the
        # internal child error pipe and the stdout pipe.
        # We also leave 10 more open for use by the Python 2 startup
        # import machinery which tends to consume several at once.
        for fd in sorted(open_fds)[:14]:
            os.close(fd)
            open_fds.remove(fd)

        for fd in open_fds:
            self.addCleanup(os.close, fd)

        max_fd_open = max(open_fds)

        import resource
        rlim_cur, rlim_max = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            # 29 is lower than the highest fds we are leaving open.
            resource.setrlimit(resource.RLIMIT_NOFILE, (29, rlim_max))
            # Launch a new Python interpreter with our low fd rlim_cur that
            # inherits open fds above that limit.  It then uses subprocess
            # with close_fds=True to get a report of open fds in the child.
            # An explicit list of fds to check is passed to fd_status.py as
            # letting fd_status rely on its default logic would miss the
            # fds above rlim_cur as it normally only checks up to that limit.
            p = subprocess.Popen(
                [sys.executable, '-c', yenv +
                 textwrap.dedent("""
                     import subprocess32, sys
                     subprocess32.Popen([sys.executable, %(fd_status)r] +
                                        [str(x) for x in range(%(max_fd)d)],
                                        close_fds=True).wait()
                     """ % dict(fd_status=fd_status, max_fd=max_fd_open+1))],
                stdout=subprocess.PIPE, close_fds=False)
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (rlim_cur, rlim_max))

        output, unused_stderr = p.communicate()
        remaining_fds = set(map(int, output.strip().split(',')))

        self.assertFalse(remaining_fds & open_fds,
                         msg="Some fds were left open.")


    def test_pass_fds(self):
        fd_status = test_support.findfile("testdata/fd_status.py")

        open_fds = set()

        for x in range(5):
            fds = os.pipe()
            self.addCleanup(os.close, fds[0])
            self.addCleanup(os.close, fds[1])
            open_fds.update(fds)

        for fd in open_fds:
            p = subprocess.Popen([sys.executable, fd_status],
                                 stdout=subprocess.PIPE, close_fds=True,
                                 pass_fds=(fd, ))
            output, ignored = p.communicate()

            remaining_fds = set(map(int, output.split(',')))
            to_be_closed = open_fds - set((fd,))

            self.assertIn(fd, remaining_fds, "fd to be passed not passed")
            self.assertFalse(remaining_fds & to_be_closed,
                             "fd to be closed passed")

            # Syntax requires Python 2.5, assertWarns requires Python 2.7.
            #with self.assertWarns(RuntimeWarning) as context:
            #    self.assertFalse(subprocess.call(
            #            [sys.executable, "-c", yenv + "import sys; sys.exit(0)"],
            #            close_fds=False, pass_fds=(fd, )))
            #self.assertIn('overriding close_fds', str(context.warning))

    def test_stdout_stdin_are_single_inout_fd(self):
        inout = open(os.devnull, "r+")
        try:
            p = subprocess.Popen([sys.executable, "-c", yenv + "import sys; sys.exit(0)"],
                                 stdout=inout, stdin=inout)
            p.wait()
        finally:
            inout.close()

    def test_stdout_stderr_are_single_inout_fd(self):
        inout = open(os.devnull, "r+")
        try:
            p = subprocess.Popen([sys.executable, "-c", yenv + "import sys; sys.exit(0)"],
                                 stdout=inout, stderr=inout)
            p.wait()
        finally:
            inout.close()

    def test_stderr_stdin_are_single_inout_fd(self):
        inout = open(os.devnull, "r+")
        try:
            p = subprocess.Popen([sys.executable, "-c", yenv + "import sys; sys.exit(0)"],
                                 stderr=inout, stdin=inout)
            p.wait()
        finally:
            inout.close()

    def test_wait_when_sigchild_ignored(self):
        # NOTE: sigchild_ignore.py may not be an effective test on all OSes.
        sigchild_ignore = test_support.findfile("testdata/sigchild_ignore.py")
        p = subprocess.Popen([sys.executable, sigchild_ignore],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        self.assertEqual(0, p.returncode, "sigchild_ignore.py exited"
                         " non-zero with this error:\n%s" % stderr)

    def test_select_unbuffered(self):
        # Issue #11459: bufsize=0 should really set the pipes as
        # unbuffered (and therefore let select() work properly).
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys;'
                              'sys.stdout.write("apple")'],
                             stdout=subprocess.PIPE,
                             bufsize=0)
        f = p.stdout
        self.addCleanup(f.close)
        try:
            self.assertEqual(f.read(4), "appl")
            self.assertIn(f, select.select([f], [], [], 0.0)[0])
        finally:
            p.wait()

    def test_zombie_fast_process_del(self):
        # Issue #12650: on Unix, if Popen.__del__() was called before the
        # process exited, it wouldn't be added to subprocess._active, and would
        # remain a zombie.
        # spawn a Popen, and delete its reference before it exits
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import sys, time;'
                              'time.sleep(0.2)'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        self.addCleanup(p.stdout.close)
        self.addCleanup(p.stderr.close)
        ident = id(p)
        pid = p.pid
        del p
        # check that p is in the active processes list
        self.assertIn(ident, [id(o) for o in subprocess._active])

    def test_leak_fast_process_del_killed(self):
        # Issue #12650: on Unix, if Popen.__del__() was called before the
        # process exited, and the process got killed by a signal, it would never
        # be removed from subprocess._active, which triggered a FD and memory
        # leak.
        # spawn a Popen, delete its reference and kill it
        p = subprocess.Popen([sys.executable, "-c", yenv +
                              'import time;'
                              'time.sleep(3)'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        self.addCleanup(p.stdout.close)
        self.addCleanup(p.stderr.close)
        ident = id(p)
        pid = p.pid
        del p
        os.kill(pid, signal.SIGKILL)
        # check that p is in the active processes list
        self.assertIn(ident, [id(o) for o in subprocess._active])

        # let some time for the process to exit, and create a new Popen: this
        # should trigger the wait() of p
        time.sleep(0.2)
        try:
            proc = subprocess.Popen(['nonexisting_i_hope'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            proc.__exit__(None, None, None)
        except EnvironmentError:
            pass
        else:
            self.fail("EnvironmentError not raised.")
        # p should have been wait()ed on, and removed from the _active list
        self.assertRaises(OSError, os.waitpid, pid, 0)
        self.assertNotIn(ident, [id(o) for o in subprocess._active])

    def test_close_fds_after_preexec(self):
        fd_status = test_support.findfile("testdata/fd_status.py")

        # this FD is used as dup2() target by preexec_fn, and should be closed
        # in the child process
        fd = os.dup(1)
        self.addCleanup(os.close, fd)

        p = subprocess.Popen([sys.executable, fd_status],
                             stdout=subprocess.PIPE, close_fds=True,
                             preexec_fn=lambda: os.dup2(1, fd))
        output, ignored = p.communicate()

        remaining_fds = set(map(int, output.split(',')))

        self.assertNotIn(fd, remaining_fds)

    def test_child_terminated_in_stopped_state(self):
        """Test wait() behavior when waitpid returns WIFSTOPPED; issue29335."""
        if not ctypes:
            sys.stderr.write('ctypes module required.\n')
            return
        if not sys.executable:
            self.stderr.write('Test requires sys.executable.\n')
            return
        PTRACE_TRACEME = 0  # From glibc and MacOS (PT_TRACE_ME).
        libc_name = ctypes.util.find_library('c')
        libc = ctypes.CDLL(libc_name)
        if not hasattr(libc, 'ptrace'):
            self.stderr.write('ptrace() required.\n')
            return
        test_ptrace = subprocess.Popen(
            [sys.executable, '-c', yenv + """if True:
             import ctypes
             libc = ctypes.CDLL({libc_name!r})
             libc.ptrace({PTRACE_TRACEME}, 0, 0)
             """.format(libc_name=libc_name, PTRACE_TRACEME=PTRACE_TRACEME)
            ])
        if test_ptrace.wait() != 0:
            sys.stderr.write('ptrace() failed - unable to test.\n')
            return
        child = subprocess.Popen(
            [sys.executable, '-c', yenv + """if True:
             import ctypes
             libc = ctypes.CDLL({libc_name!r})
             libc.ptrace({PTRACE_TRACEME}, 0, 0)
             libc.printf(ctypes.c_char_p(0xdeadbeef))  # Crash the process.
             """.format(libc_name=libc_name, PTRACE_TRACEME=PTRACE_TRACEME)
            ])
        try:
            returncode = child.wait()
        except Exception, e:
            child.kill()  # Clean up the hung stopped process.
            raise e
        self.assertNotEqual(0, returncode)
        self.assert_(returncode < 0, msg=repr(returncode))  # signal death, likely SIGSEGV.


if mswindows:
    class POSIXProcessTestCase(unittest.TestCase): pass


#@unittest.skipUnless(mswindows, "Windows specific tests")
class Win32ProcessTestCase(BaseTestCase):

    def test_startupinfo(self):
        # startupinfo argument
        # We uses hardcoded constants, because we do not want to
        # depend on win32all.
        STARTF_USESHOWWINDOW = 1
        SW_MAXIMIZE = 3
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags = STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = SW_MAXIMIZE
        # Since Python is a console process, it won't be affected
        # by wShowWindow, but the argument should be silently
        # ignored
        subprocess.call([sys.executable, "-c", yenv + "import sys; sys.exit(0)"],
                        startupinfo=startupinfo)

    def test_creationflags(self):
        # creationflags argument
        CREATE_NEW_CONSOLE = 16
        sys.stderr.write("    a DOS box should flash briefly ...\n")
        subprocess.call(sys.executable +
                        ' -c "import time; time.sleep(0.25)"',
                        creationflags=CREATE_NEW_CONSOLE)

    def test_invalid_args(self):
        # invalid arguments should raise ValueError
        self.assertRaises(ValueError, subprocess.call,
                          [sys.executable, "-c", yenv +
                           "import sys; sys.exit(47)"],
                          preexec_fn=lambda: 1)
        self.assertRaises(ValueError, subprocess.call,
                          [sys.executable, "-c", yenv +
                           "import sys; sys.exit(47)"],
                          stdout=subprocess.PIPE,
                          close_fds=True)

    def test_close_fds(self):
        # close file descriptors
        rc = subprocess.call([sys.executable, "-c", yenv +
                              "import sys; sys.exit(47)"],
                              close_fds=True)
        self.assertEqual(rc, 47)

    def test_shell_sequence(self):
        # Run command through the shell (sequence)
        newenv = os.environ.copy()
        newenv["FRUIT"] = "physalis"
        p = subprocess.Popen(["set"], shell=1,
                             stdout=subprocess.PIPE,
                             env=newenv)
        self.assertIn("physalis", p.stdout.read())

    def test_shell_string(self):
        # Run command through the shell (string)
        newenv = os.environ.copy()
        newenv["FRUIT"] = "physalis"
        p = subprocess.Popen("set", shell=1,
                             stdout=subprocess.PIPE,
                             env=newenv)
        self.assertIn("physalis", p.stdout.read())

    def test_call_string(self):
        # call() function with string argument on Windows
        rc = subprocess.call(sys.executable +
                             ' -c "import sys; sys.exit(47)"')
        self.assertEqual(rc, 47)

    def _kill_process(self, method, *args):
        # Some win32 buildbot raises EOFError if stdin is inherited
        p = subprocess.Popen([sys.executable, "-c", yenv + "input()"],
                             stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # Let the process initialize (Issue #3137)
        time.sleep(0.1)
        # The process should not terminate prematurely
        self.assert_(p.poll() is None)
        # Retry if the process do not receive the signal.
        count, maxcount = 0, 3
        while count < maxcount and p.poll() is None:
            getattr(p, method)(*args)
            time.sleep(0.1)
            count += 1

        returncode = p.poll()
        self.assert_(returncode is not None, "the subprocess did not terminate")
        if count > 1:
            print >>sys.stderr, ("p.{}{} succeeded after "
                                 "{} attempts".format(method, args, count))
        _, stderr = p.communicate()
        self.assertStderrEqual(stderr, '')
        self.assertEqual(p.wait(), returncode)
        self.assertNotEqual(returncode, 0)

    def test_send_signal(self):
        self._kill_process('send_signal', signal.SIGTERM)

    def test_kill(self):
        self._kill_process('kill')

    def test_terminate(self):
        self._kill_process('terminate')


if not mswindows:
    class Win32ProcessTestCase(unittest.TestCase): pass


#@unittest.skipUnless(getattr(subprocess, '_has_poll', False),
#                     "poll system call not supported")
class ProcessTestCaseNoPoll(ProcessTestCase):
    def setUp(self):
        subprocess._has_poll = False
        ProcessTestCase.setUp(self)

    def tearDown(self):
        subprocess._has_poll = True
        ProcessTestCase.tearDown(self)


if not getattr(subprocess, '_has_poll', False):
    class ProcessTestCaseNoPoll(unittest.TestCase): pass


#@unittest.skipUnless(getattr(subprocess, '_posixsubprocess', False),
#                     "_posixsubprocess extension module not found.")
class ProcessTestCasePOSIXPurePython(ProcessTestCase, POSIXProcessTestCase):
    def setUp(self):
        subprocess._posixsubprocess = None
        ProcessTestCase.setUp(self)
        POSIXProcessTestCase.setUp(self)

    def tearDown(self):
        subprocess._posixsubprocess = sys.modules['_posixsubprocess32']
        POSIXProcessTestCase.tearDown(self)
        ProcessTestCase.tearDown(self)


class POSIXSubprocessModuleTestCase(unittest.TestCase):
    def test_fork_exec_sorted_fd_sanity_check(self):
        # Issue #23564: sanity check the fork_exec() fds_to_keep sanity check.
        _posixsubprocess = subprocess._posixsubprocess
        gc_enabled = gc.isenabled()
        try:
            gc.enable()

            for fds_to_keep in (
                (-1, 2, 3, 4, 5),  # Negative number.
                ('str', 4),  # Not an int.
                (18, 23, 42, 2**63),  # Out of range.
                (5, 4),  # Not sorted.
                (6, 7, 7, 8),  # Duplicate.
            ):
                try:
                    _posixsubprocess.fork_exec(
                        ["false"], ["false"],
                        True, fds_to_keep, None, ["env"],
                        -1, -1, -1, -1,
                        1, 2, 3, 4,
                        True, True, None)
                except ValueError, exception:
                    self.assertTrue('fds_to_keep' in str(exception),
                                    msg=str(exception))
                else:
                    self.fail("ValueError not raised, fds_to_keep=%s" %
                              (fds_to_keep,))
        finally:
            if not gc_enabled:
                gc.disable()

    def test_cloexec_pass_fds(self):
        if not os.path.exists('/dev/null') or not os.path.isdir('/dev/fd'):
            print("Skipped - This test requires /dev/null and /dev/fd/*.")
            return
        null_reader_proc = subprocess.Popen(
                ["cat"],
                stdin=open('/dev/null', 'rb'),
                stdout=subprocess.PIPE)
        try:
            data = null_reader_proc.stdout
            fd_name = '/dev/fd/%d' % data.fileno()
            fd_reader_proc = subprocess.Popen(
                    ["cat", fd_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Capture any error from cat.
                    pass_fds=(data.fileno(),))
            try:
                fddata = fd_reader_proc.stdout
                self.assertEqual('', fddata.read())
            finally:
                fd_reader_proc.wait()
        finally:
            null_reader_proc.wait()


if not getattr(subprocess, '_posixsubprocess', False):
    print >>sys.stderr, "_posixsubprocess extension module not found."
    class ProcessTestCasePOSIXPurePython(unittest.TestCase): pass
    class POSIXSubprocessModuleTestCase(unittest.TestCase): pass


class HelperFunctionTests(unittest.TestCase):
    #@unittest.skipIf(mswindows, "errno and EINTR make no sense on windows")
    def test_eintr_retry_call(self):
        record_calls = []
        def fake_os_func(*args):
            record_calls.append(args)
            if len(record_calls) == 2:
                raise OSError(errno.EINTR, "fake interrupted system call")
            return tuple(reversed(args))

        self.assertEqual((999, 256),
                         subprocess._eintr_retry_call(fake_os_func, 256, 999))
        self.assertEqual([(256, 999)], record_calls)
        # This time there will be an EINTR so it will loop once.
        self.assertEqual((666,),
                         subprocess._eintr_retry_call(fake_os_func, 666))
        self.assertEqual([(256, 999), (666,), (666,)], record_calls)

    if mswindows:
        del test_eintr_retry_call

    if not hasattr(unittest.TestCase, 'assertSequenceEqual'):
        def assertSequenceEqual(self, seq1, seq2):
            self.assertEqual(list(seq1), list(seq2))

    def test_get_exec_path(self):
        defpath_list = os.defpath.split(os.pathsep)
        test_path = ['/monty', '/python', '', '/flying/circus']
        test_env = {'PATH': os.pathsep.join(test_path)}

        get_exec_path = subprocess._get_exec_path
        saved_environ = os.environ
        try:
            os.environ = dict(test_env)
            # Test that defaulting to os.environ works.
            self.assertSequenceEqual(test_path, get_exec_path())
            self.assertSequenceEqual(test_path, get_exec_path(env=None))
        finally:
            os.environ = saved_environ

        # No PATH environment variable
        self.assertSequenceEqual(defpath_list, get_exec_path({}))
        # Empty PATH environment variable
        self.assertSequenceEqual(('',), get_exec_path({'PATH':''}))
        # Supplied PATH environment variable
        self.assertSequenceEqual(test_path, get_exec_path(test_env))

    def test_args_from_interpreter_flags(self):
        if sys.version_info[:2] < (2,6):
            print "Skipped - only useful on 2.6 and higher."
            return
        # Mostly just to call it for code coverage.
        args_list = subprocess32._args_from_interpreter_flags()
        self.assertTrue(isinstance(args_list, list), msg=repr(args_list))

    def test_timeout_expired_unpickling(self):
        """https://github.com/google/python-subprocess32/issues/57"""
        t = subprocess32.TimeoutExpired(['command', 'arg1'], 5,
                                        output='stdout!', stderr='err')
        t_pickled = pickle.dumps(t)
        t2 = pickle.loads(t_pickled)
        self.assertEqual(t.cmd, t2.cmd)
        self.assertEqual(t.timeout, t2.timeout)
        self.assertEqual(t.output, t2.output)
        self.assertEqual(t.stderr, t2.stderr)

    def test_called_process_error_unpickling(self):
        """https://github.com/google/python-subprocess32/issues/57"""
        e = subprocess32.CalledProcessError(
                2, ['command', 'arg1'], output='stdout!', stderr='err')
        e_pickled = pickle.dumps(e)
        e2 = pickle.loads(e_pickled)
        self.assertEqual(e.returncode, e2.returncode)
        self.assertEqual(e.cmd, e2.cmd)
        self.assertEqual(e.output, e2.output)
        self.assertEqual(e.stderr, e2.stderr)


def reap_children():
    """Use this function at the end of test_main() whenever sub-processes
    are started.  This will help ensure that no extra children (zombies)
    stick around to hog resources and create problems when looking
    for refleaks.
    """

    # Reap all our dead child processes so we don't leave zombies around.
    # These hog resources and might be causing some of the buildbots to die.
    if hasattr(os, 'waitpid'):
        any_process = -1
        while True:
            try:
                # This will raise an exception on Windows.  That's ok.
                pid, status = os.waitpid(any_process, os.WNOHANG)
                if pid == 0:
                    break
            except:
                break



class ContextManagerTests(BaseTestCase):

    def test_pipe(self):
        proc = subprocess.Popen([sys.executable, "-c", yenv +
                               "import sys;"
                               "sys.stdout.write('stdout');"
                               "sys.stderr.write('stderr');"],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        try:
            self.assertEqual(proc.stdout.read(), "stdout")
            self.assertStderrEqual(proc.stderr.read(), "stderr")
        finally:
            proc.__exit__(None, None, None)

        self.assertTrue(proc.stdout.closed)
        self.assertTrue(proc.stderr.closed)

    def test_returncode(self):
        proc = subprocess.Popen([sys.executable, "-c", yenv +
                               "import sys; sys.exit(100)"])
        proc.__exit__(None, None, None)
        # __exit__ calls wait(), so the returncode should be set
        self.assertEqual(proc.returncode, 100)

    def test_communicate_stdin(self):
        proc = subprocess.Popen([sys.executable, "-c", yenv +
                              "import sys;"
                              "sys.exit(sys.stdin.read() == 'context')"],
                             stdin=subprocess.PIPE)
        try:
            proc.communicate("context")
            self.assertEqual(proc.returncode, 1)
        finally:
            proc.__exit__(None, None, None)

    def test_invalid_args(self):
        try:
            proc = subprocess.Popen(['nonexisting_i_hope'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            proc.__exit__(None, None, None)
        except EnvironmentError, exception:
            # ignore errors that indicate the command was not found
            if exception.errno not in (errno.ENOENT, errno.EACCES):
                raise
        else:
            self.fail("Expected an EnvironmentError exception.")


if sys.version_info[:2] <= (2,4):
    # The test suite hangs during the pure python test on 2.4.  No idea why.
    # That is not the implementation anyone is using this module for anyways.
    class ProcessTestCasePOSIXPurePython(unittest.TestCase): pass


def main():
    unit_tests = (ProcessTestCase,
                  POSIXProcessTestCase,
                  POSIXSubprocessModuleTestCase,
                  Win32ProcessTestCase,
                  ProcessTestCasePOSIXPurePython,
                  ProcessTestCaseNoPoll,
                  HelperFunctionTests,
                  ContextManagerTests,
                  RunFuncTestCase,
                 )

    test_support.run_unittest(*unit_tests)
    reap_children()

if __name__ == "__main__":
    main()
