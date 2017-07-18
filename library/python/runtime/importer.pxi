import sys
import imp
import importlib
import marshal
import traceback
import __res as __resource
import os

env_entry_point = 'Y_PYTHON_ENTRY_POINT'
env_source_root = 'Y_PYTHON_SOURCE_ROOT'
executable = sys.executable
sys.modules['run_import_hook'] = __resource


def iter_keys():
    for idx in range(__resource.count()):
        key = __resource.key_by_index(idx)

        if key.startswith('/py_modules/'):
            mod = key[12:]
            if '/' in mod:
                raise Exception('Incorrect py_modules resource: ' + repr(key))
            yield mod


def iter_prefixes(s):
    i = s.find('.')
    while i >= 0:
        yield s[:i]
        i = s.find('.', i + 1)


class ResourceImporter(object):

    """ A meta_path importer that loads code from built-in resources.
    """

    def __init__(self):
        self.memory = set(list(iter_keys()))  # Set of importable module names.
        self.source_map = {}                  # Map from file names to module names.
        self._source_name = {}                # Map from original to altered module names.
        self._source_root = os.environ.get(env_source_root, None)
        if self._source_root is not None:
            self._source_root = os.path.abspath(os.path.expanduser(self._source_root))
            os.environ[env_source_root] = self._source_root

        for p in list(self.memory) + list(sys.builtin_module_names):
            for pp in iter_prefixes(p):
                k = pp + '.__init__'
                if k not in self.memory:
                    self.memory.add(k)

    def contains(self, name):
        """See if a module or package is in the dict."""
        if name in self.memory:
            return name
        package_name = '{0}.__init__'.format(name)
        if package_name in self.memory:
            return package_name
        return False

    __contains__ = contains  # Convenience.

    def find_module(self, fullname, path=None):
        """Find the module in the dict."""
        if fullname in self:
            return self
        return None

    def source_path(self, fullname):
        """Return the module name if the module is in the dict."""
        if fullname not in self:
            raise ImportError
        return fullname

    def file_source(self, filename):
        if not self.source_map:
            for i in xrange(__resource.count()):
                key = __resource.key_by_index(i)
                if key.startswith('/py_fs/'):
                    modname = key[len('/py_fs/'):]
                    path = __resource.find(key)
                    self.source_map[path] = '/py_modules/' + modname

        return self.source_map.get(filename, '')

    _get_resource = __resource.find

    def _get_data(self, path, prefix):
        """Return the bytes for the source.

        The value found in the dict is passed through 'bytes' before being
        returned.

        """
        name = self.contains(path)
        if not name:
            raise IOError
        res = self._get_resource(prefix + name)
        if res is None:
            # XXX(borman) A resource is missing? Let's just pretend it is empty.
            return ''
        return bytes(res)

    def get_data(self, path):
        return self._get_data(path, '/py_modules/')

    # Used by `linecache` (while printing tracebacks) unless module filename
    # exists on the filesystem.
    def get_source(self, mod_name):
        return self.get_data(self._source_name.get(mod_name, mod_name))

    def get_code(self, mod_name):
        if self._source_root:
            key = '/py_fs/' + mod_name
            path = __resource.find(key)
            if not path:
                path = __resource.find(key + '.__init__')
            if path:
                path = os.path.join(self._source_root, path)
                if os.path.exists(path):
                    with open(path) as f:
                        return compile(f.read(), path, 'exec', dont_inherit=True)

        pyc = self._get_data(mod_name, '/py_code/')
        if pyc:
            return marshal.loads(pyc)
        else:
            # This covers packages with no __init__.py in resources.
            return compile('', mod_name, 'exec', dont_inherit=True)

    def is_package(self, fullname):
        if fullname in self.memory:
            return False

        if fullname + '.__init__' in self.memory:
            return True

        raise ImportError

    def load_module(self, mod_name, fix_name=None):
        code = self.get_code(mod_name)
        is_package = self.is_package(mod_name)
        source_name = self._source_name

        mod = imp.new_module(mod_name)
        mod.__loader__ = self
        mod.__file__ = code.co_filename

        if is_package:
            mod.__path__ = [executable]
            mod.__package__ = mod_name
        else:
            mod.__package__ = mod_name.rpartition('.')[0]

        sys.modules[mod_name] = mod

        if fix_name:
            mod.__name__ = fix_name
            self._source_name = dict(source_name, **{fix_name: mod_name})

        exec code in mod.__dict__

        if fix_name:
            mod.__name__ = mod_name
            self._source_name = source_name

        # Some hacky modules (e.g. pygments.lexers) replace themselves in
        # `sys.modules` with proxies.
        return sys.modules[mod_name]

    def run_main(self):
        entry_point = os.environ.pop(env_entry_point, None)

        if entry_point is None:
            entry_point = __resource.find('PY_MAIN')

        if entry_point is None:
            entry_point = '__main__'

        if ':' in entry_point:
            mod_name, func_name = entry_point.split(':', 1)
            if mod_name == '':
                mod_name = 'library.python.runtime.entry_points'
            mod = importlib.import_module(mod_name)
            func = getattr(mod, func_name)
            return func()

        if entry_point not in self.memory:
            raise Exception(entry_point + ' not found')

        self.load_module(entry_point, fix_name='__main__')


importer = ResourceImporter()
sys.meta_path.insert(0, importer)


def executable_path_hook(path):
    if path == executable:
        return importer

    raise ImportError


sys.path.insert(0, executable)
sys.path_hooks.insert(0, executable_path_hook)

# Indicator that modules and resources are built-in rather than on the file system.
sys.is_standalone_binary = True
sys.frozen = True

# Set of names of importable modules.
sys.extra_modules = importer.memory

# Use pure-python implementation of traceback printer.
# Built-in printer (PyTraceBack_Print) does not support custom module loaders
sys.excepthook = traceback.print_exception
