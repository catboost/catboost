import copy
import imp
import importlib
import marshal
import os
import re
import sys
import traceback
from os.path import sep as path_sep

import __res as __resource

env_entry_point = 'Y_PYTHON_ENTRY_POINT'
env_source_root = 'Y_PYTHON_SOURCE_ROOT'
executable = sys.executable or 'Y_PYTHON'
sys.modules['run_import_hook'] = __resource

find_py_module = lambda mod: __resource.find('/py_modules/' + mod)
find_py_code = lambda mod: __resource.find('/py_code/' + mod)

Y_PYTHON_SOURCE_ROOT = os.environ.get(env_source_root)
if Y_PYTHON_SOURCE_ROOT is not None:
    Y_PYTHON_SOURCE_ROOT = os.path.abspath(os.path.expanduser(Y_PYTHON_SOURCE_ROOT))
    os.environ[env_source_root] = Y_PYTHON_SOURCE_ROOT


def file_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


def iter_keys(prefix):
    l = len(prefix)
    for idx in xrange(__resource.count()):
        key = __resource.key_by_index(idx)
        if key.startswith(prefix):
            yield key, key[l:]


def iter_py_modules(with_keys=False):
    for key, mod in iter_keys('/py_modules/'):
        if '/' in mod:
            raise Exception('Incorrect py_modules resource: ' + repr(key))
        if with_keys:
            yield key, mod
        else:
            yield mod


def iter_prefixes(s):
    i = s.find('.')
    while i >= 0:
        yield s[:i]
        i = s.find('.', i + 1)


def resfs_resolve(path):
    """
    Return the absolute path of a root-relative path if it exists.
    """
    if Y_PYTHON_SOURCE_ROOT:
        abspath = os.path.join(Y_PYTHON_SOURCE_ROOT, path)
        if os.path.exists(abspath):
            return abspath


def resfs_src(key, resfs_file=False):
    """
    Return the root-relative file path of a resource key.
    """
    if resfs_file:
        key = 'resfs/file/' + key
    return __resource.find('resfs/src/' + key)


def resfs_read(path, builtin=None):
    """
    Return the bytes of the resource file at path, or None.
    If builtin is True, do not look for it on the filesystem.
    If builtin is False, do not look in the builtin resources.
    """
    if builtin is not True:
        arcpath = resfs_src(path, resfs_file=True)
        if arcpath:
            fspath = resfs_resolve(arcpath)
            if fspath:
                return file_bytes(fspath)

    if builtin is not False:
        return __resource.find('resfs/file/' + path)


def resfs_files(prefix=''):
    """
    List builtin resource file paths.
    """
    return [key[11:] for key, _ in iter_keys('resfs/file/' + prefix)]


class ResourceImporter(object):

    """ A meta_path importer that loads code from built-in resources.
    """

    def __init__(self):
        self.memory = set(iter_py_modules())  # Set of importable module names.
        self.source_map = {}                  # Map from file names to module names.
        self._source_name = {}                # Map from original to altered module names.
        self._package_prefix = ''

        self._before_import_callback = None
        self._after_import_callback = None

        for p in list(self.memory) + list(sys.builtin_module_names):
            for pp in iter_prefixes(p):
                k = pp + '.__init__'
                if k not in self.memory:
                    self.memory.add(k)

    def for_package(self, name):
        importer = copy.copy(self)
        importer._package_prefix = name + '.'
        return importer

    def set_callbacks(self, before_import=None, after_import=None):
        """Callable[[module], None]"""
        self._before_import_callback = before_import
        self._after_import_callback = after_import

    # PEP-302 finder.
    def find_module(self, fullname, path=None):
        try:
            self.is_package(fullname)
        except ImportError:
            return None
        return self

    # PEP-302 extension 1 of 3: data loader.
    def get_data(self, path):
        abspath = resfs_resolve(path)
        if abspath:
            return file_bytes(abspath)
        path = path.replace('\\', '/')
        data = resfs_read(path, builtin=True)
        if data is None:
            raise IOError(path)  # Y_PYTHON_ENTRY_POINT=:resource_files
        return data

    # PEP-302 extension 2 of 3: get __file__ without importing.
    def get_filename(self, fullname):
        modname = fullname
        if self.is_package(fullname):
            fullname += '.__init__'
        return resfs_src('/py_modules/' + fullname) or modname

    # PEP-302 extension 3 of 3: packaging introspection.
    # Used by `linecache` (while printing tracebacks) unless module filename
    # exists on the filesystem.
    def get_source(self, fullname):
        fullname = self._source_name.get(fullname, fullname)
        if self.is_package(fullname):
            fullname += '.__init__'

        abspath = resfs_resolve(self.get_filename(fullname))
        if abspath:
            return file_bytes(abspath)
        return find_py_module(fullname)

    def get_code(self, fullname):
        modname = fullname
        if self.is_package(fullname):
            fullname += '.__init__'

        abspath = resfs_resolve(self.get_filename(fullname))
        if abspath:
            data = file_bytes(abspath)
            return compile(data, abspath, 'exec', dont_inherit=True)

        pyc = find_py_code(fullname)
        if pyc:
            return marshal.loads(pyc)
        else:
            # This covers packages with no __init__.py in resources.
            return compile('', modname, 'exec', dont_inherit=True)

    def is_package(self, fullname):
        if fullname in self.memory:
            return False

        if fullname + '.__init__' in self.memory:
            return True

        raise ImportError(fullname)

    # Extension for contrib/python/coverage.
    def file_source(self, filename):
        """
        Return the key of the module source by its resource path.
        """
        if not self.source_map:
            for key, mod in iter_py_modules(with_keys=True):
                path = self.get_filename(mod)
                self.source_map[path] = key

        if filename in self.source_map:
            return self.source_map[filename]

        if resfs_read(filename, builtin=True) is not None:
            return 'resfs/file/' + filename

        return ''

    # Extension for pkgutil.iter_modules.
    def iter_modules(self, prefix=''):
        rx = re.compile(re.escape(self._package_prefix) + r'([^.]+)(\.__init__)?$')
        for p in self.memory:
            m = rx.match(p)
            if m:
                yield prefix + m.group(1), m.group(2) is not None

    # PEP-302 loader.
    def load_module(self, mod_name, fix_name=None):
        code = self.get_code(mod_name)
        is_package = self.is_package(mod_name)
        source_name = self._source_name

        mod = imp.new_module(mod_name)
        mod.__loader__ = self
        mod.__file__ = code.co_filename

        if is_package:
            mod.__path__ = [executable + path_sep + mod_name.replace('.', path_sep)]
            mod.__package__ = mod_name
        else:
            mod.__package__ = mod_name.rpartition('.')[0]

        if fix_name:
            mod.__name__ = fix_name
            self._source_name = dict(source_name, **{fix_name: mod_name})

        old_mod = sys.modules.get(mod_name, None)
        sys.modules[mod_name] = mod

        # __name__ and __file__ could be overwritten after execution
        # So these two things are needed if wee want to be consistent at some point
        initial_modname = mod.__name__
        initial_filename = mod.__file__

        if self._before_import_callback:
            self._before_import_callback(initial_modname, initial_filename)

        try:
            exec code in mod.__dict__
            old_mod = sys.modules[mod_name]
        finally:
            sys.modules[mod_name] = old_mod

            # "Zero-cost". Just in case import error occurs
            if self._after_import_callback:
                self._after_import_callback(initial_modname, initial_filename)

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

    if path.startswith(executable + path_sep):
        return importer.for_package(path[len(executable + path_sep):].replace(path_sep, '.'))

    raise ImportError(path)


if executable not in sys.path:
    sys.path.insert(0, executable)
sys.path_hooks.insert(0, executable_path_hook)
sys.path_importer_cache[executable] = importer

# Indicator that modules and resources are built-in rather than on the file system.
sys.is_standalone_binary = True
sys.frozen = True

# Set of names of importable modules.
sys.extra_modules = importer.memory

# Use pure-python implementation of traceback printer.
# Built-in printer (PyTraceBack_Print) does not support custom module loaders
sys.excepthook = traceback.print_exception
