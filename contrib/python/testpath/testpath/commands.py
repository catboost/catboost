import contextlib
import json
import os
import shutil
import sys
import tempfile

__all__ = ['MockCommand', 'assert_calls']

pkgdir = os.path.dirname(__file__)

recording_dir = None

def prepend_to_path(dir):
    os.environ['PATH'] = dir + os.pathsep + os.environ['PATH']

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
"""

# TODO: Overlapping calls to the same command may interleave writes.

class MockCommand(object):
    """Context manager to mock a system command.
    
    The mock command will be written to a directory at the front of $PATH,
    taking precedence over any existing command with the same name.
    
    By specifying content as a string, you can determine what running the
    command will do. The default content records each time the command is
    called and exits: you can access these records with mockcmd.get_calls().
    
    On Windows, the specified content will be run by the Python interpreter in
    use. On Unix, it should start with a shebang (``#!/path/to/interpreter``).
    """
    def __init__(self, name, content=None):
        global recording_dir
        self.name = name
        self.content = content

        if recording_dir is None:
            recording_dir = tempfile.mkdtemp()
        fd, self.recording_file = tempfile.mkstemp(dir=recording_dir,
                                                prefix=name, suffix='.json')
        os.close(fd)
        self.command_dir = tempfile.mkdtemp()

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
        
        if self.content is None:
            self.content = _record_run.format(python=sys.executable,
                                             recording_file=self.recording_file)

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
        
        This relies on the default script content, so it will return an
        empty list if you specified a different content parameter.
        
        For each time the command was run, the list will contain a dictionary
        with keys argv, env and cwd.
        """
        if recording_dir is None:
            return []
        if not os.path.isfile(self.recording_file):
            return []
        
        with open(self.recording_file, 'r') as f:
            # 1E is ASCII record separator, last chunk is empty
            chunks = f.read().split('\x1e')[:-1]
        
        return [json.loads(c) for c in chunks]


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
    
    calls = mc.get_calls()
    assert calls != [], "Command %r was not called" % cmd

    if args is not None:
        if not any(args == c['argv'][1:] for c in calls):
            msg = ["Command %r was not called with specified args (%r)" %
                            (cmd, args),
                   "It was called with these arguments: "]
            for c in calls:
                msg.append('  %r' % c['argv'][1:])
            raise AssertionError('\n'.join(msg))
