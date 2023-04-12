import contextlib
import json
import os
import shutil
import sys
import tempfile

__all__ = ['MockCommand', 'assert_calls']

pkgdir = os.path.dirname(__file__)

recording_dir = None

def _make_recording_file(prefix):
    """Make a temp file for recording calls to a mocked command"""
    global recording_dir
    if recording_dir is None:
        recording_dir = tempfile.mkdtemp()
    fd, p = tempfile.mkstemp(dir=recording_dir, prefix=prefix, suffix='.json')
    os.close(fd)
    return p

def prepend_to_path(dir):
    os.environ['PATH'] = dir + os.pathsep + os.environ.get('PATH', os.defpath)

def remove_from_path(dir):
    path_dirs = os.environ['PATH'].split(os.pathsep)
    path_dirs.remove(dir)
    os.environ['PATH'] = os.pathsep.join(path_dirs)


_record_run = """#!{python}
import os, sys
import json

with open({recording_file!r}, 'a') as f:
    json.dump({{'env': dict(os.environ),
               'argv': sys.argv,
               'cwd': os.getcwd()}},
              f)
    f.write('\\x1e') # ASCII record separator

{extra_code}
"""

_output_template = """\
sys.stdout.write({!r})
sys.stderr.write({!r})
sys.exit({!r})
"""

# TODO: Overlapping calls to the same command may interleave writes.

class MockCommand(object):
    """Context manager to mock a system command.
    
    The mock command will be written to a directory at the front of $PATH,
    taking precedence over any existing command with the same name.

    The *python* parameter accepts a string of code for the command to run,
    in addition to the default behaviour of recording calls to the command.
    This will run with the same Python interpreter as the calling code, but in
    a new process.

    The *content* parameter gives extra control, by providing a script which
    will run with no additions. On Unix, it should start with a shebang (e.g.
    ``#!/usr/bin/env python``) specifying the interpreter. On Windows, it will
    always be run by the same Python interpreter as the calling code.
    Calls to the command will not be recorded when content is specified.
    """
    def __init__(self, name, content=None, python=''):
        self.name = name
        self.recording_file = _make_recording_file(prefix=name)
        self.command_dir = tempfile.mkdtemp()

        if content is None:
            content = _record_run.format(
                python=sys.executable, recording_file=self.recording_file,
                extra_code=python,
            )
        elif python:
            raise ValueError(
                "Specify script content or extra code (python='...'), not both"
            )
        self.content = content

    @classmethod
    def fixed_output(cls, name, stdout='', stderr='', exit_status=0):
        """Make a mock command, producing fixed output when it is run::

            t = 'Sat 24 Apr 17:11:58 BST 2021\\n'
            with MockCommand.fixed_output('date', t) as mock_date:
                ...

        The stdout & stderr strings will be written to the respective streams,
        and the process will exit with the specified numeric status (the default
        of 0 indicates success).

        This works with the recording mechanism, so you can check what arguments
        this command was called with.
        """
        return cls(
            name, python=_output_template.format(stdout, stderr, exit_status)
        )

    def _copy_exe(self):
        bitness = '64' if (sys.maxsize > 2**32) else '32'
        src = os.path.join(pkgdir, 'cli-%s.exe' % bitness)
        dst = os.path.join(self.command_dir, self.name+'.exe')
        shutil.copy(src, dst)

    @property
    def _cmd_path(self):
        # Can only be used once commands_dir has been set
        p = os.path.join(self.command_dir, self.name)
        if os.name == 'nt':
            p += '-script.py'
        return p

    def __enter__(self):
        if os.path.isfile(self._cmd_path):
            raise EnvironmentError("Command %r already exists at %s" %
                                            (self.name, self._cmd_path))

        with open(self._cmd_path, 'w') as f:
            f.write(self.content)
        
        if os.name == 'nt':
            self._copy_exe()
        else:
            os.chmod(self._cmd_path, 0o755) # Set executable bit

        prepend_to_path(self.command_dir)

        return self
    
    def __exit__(self, etype, evalue, tb):
        remove_from_path(self.command_dir)
        shutil.rmtree(self.command_dir, ignore_errors=True)

    def get_calls(self):
        """Get a list of calls made to this mocked command.
        
        For each time the command was run, the list will contain a dictionary
        with keys argv, env and cwd.

        This won't work if you used the *content* parameter to alter what
        the mocked command does.
        """
        if recording_dir is None:
            return []
        if not os.path.isfile(self.recording_file):
            return []
        
        with open(self.recording_file, 'r') as f:
            # 1E is ASCII record separator, last chunk is empty
            chunks = f.read().split('\x1e')[:-1]
        
        return [json.loads(c) for c in chunks]

    def assert_called(self, args=None):
        """Assert that the mock command has been called at least once.

        If args is passed, also check that it was called at least once with the
        given arguments (not including the command name), e.g.::

            with MockCommand('rsync') as mock_rsync:
                function_to_test()

            mock_rsync.assert_called(['/var/log', 'backup-server:logs'])

        This won't work if you used the *content* parameter to alter what
        the mocked command does.
        """
        calls = self.get_calls()
        assert calls != [], "Command %r was not called" % self.name

        if args is not None:
            if not any(args == c['argv'][1:] for c in calls):
                msg = ["Command %r was not called with specified args (%r)" %
                       (self.name, args),
                       "It was called with these arguments: "]
                for c in calls:
                    msg.append('  %r' % c['argv'][1:])
                raise AssertionError('\n'.join(msg))


@contextlib.contextmanager
def assert_calls(cmd, args=None):
    """Assert that a block of code runs the given command.
    
    If args is passed, also check that it was called at least once with the
    given arguments (not including the command name).
    
    Use as a context manager, e.g.::
    
        with assert_calls('git'):
            some_function_wrapping_git()
            
        with assert_calls('git', ['add', myfile]):
            some_other_function()
    """
    with MockCommand(cmd) as mc:
        yield
    
    mc.assert_called(args=args)
