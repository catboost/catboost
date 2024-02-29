import marshal
import sys
from _codecs import utf_8_decode, utf_8_encode
from _frozen_importlib import _call_with_frames_removed, spec_from_loader, BuiltinImporter
from _frozen_importlib_external import _os, _path_isfile, _path_isabs, path_sep, _path_join, _path_split
from _io import FileIO

import __res as __resource

_b = lambda x: x if isinstance(x, bytes) else utf_8_encode(x)[0]
_s = lambda x: x if isinstance(x, str) else utf_8_decode(x)[0]
env_source_root = b'Y_PYTHON_SOURCE_ROOT'
cfg_source_root = b'arcadia-source-root'
env_extended_source_search = b'Y_PYTHON_EXTENDED_SOURCE_SEARCH'
res_ya_ide_venv = b'YA_IDE_VENV'
executable = sys.executable or 'Y_PYTHON'
sys.modules['run_import_hook'] = __resource

def _probe(environ_dict, key, default_value=None):
    """ Probe bytes and str variants for environ.
    This is because in python3:
    * _os (nt) on windows returns str,
    * _os (posix) on linux return bytes
    For more information check:
    * https://github.com/python/cpython/blob/main/Lib/importlib/_bootstrap_external.py#L34
    * YA-1700
    """
    keys = [_b(key), _s(key)]
    for key in keys:
        if key in environ_dict:
            return _b(environ_dict[key])

    return _b(default_value) if isinstance(default_value, str) else default_value

# This is the prefix in contrib/tools/python3/Lib/ya.make.
py_prefix = b'py/'
py_prefix_len = len(py_prefix)

YA_IDE_VENV = __resource.find(res_ya_ide_venv)
Y_PYTHON_EXTENDED_SOURCE_SEARCH = _probe(_os.environ, env_extended_source_search) or YA_IDE_VENV


def _init_venv():
    if not _path_isabs(executable):
        raise RuntimeError(f'path in sys.executable is not absolute: {executable}')

    # Creative copy-paste from site.py
    exe_dir, _ = _path_split(executable)
    site_prefix, _ = _path_split(exe_dir)
    libpath = _path_join(site_prefix, 'lib',
                           'python%d.%d' % sys.version_info[:2],
                           'site-packages')
    sys.path.insert(0, libpath)

    # emulate site.venv()
    sys.prefix = site_prefix
    sys.exec_prefix = site_prefix

    conf_basename = 'pyvenv.cfg'
    candidate_confs = [
        conffile for conffile in (
            _path_join(exe_dir, conf_basename),
            _path_join(site_prefix, conf_basename)
            )
        if _path_isfile(conffile)
        ]
    if not candidate_confs:
        raise RuntimeError(f'{conf_basename} not found')
    virtual_conf = candidate_confs[0]
    with FileIO(virtual_conf, 'r') as f:
        for line in f:
            if b'=' in line:
                key, _, value = line.partition(b'=')
                key = key.strip().lower()
                value = value.strip()
                if key == cfg_source_root:
                    return value
    raise RuntimeError(f'{cfg_source_root} key not found in {virtual_conf}')


def _get_source_root():
    env_value = _probe(_os.environ, env_source_root)
    if env_value or not YA_IDE_VENV:
        return env_value

    return _init_venv()


Y_PYTHON_SOURCE_ROOT = _get_source_root()


def _print(*xs):
    """
    This is helpful for debugging, since automatic bytes to str conversion is
    not available yet.  It is also possible to debug with GDB by breaking on
    __Pyx_AddTraceback (with Python GDB pretty printers enabled).
    """
    parts = []
    for s in xs:
        if not isinstance(s, (bytes, str)):
            s = str(s)
        parts.append(_s(s))
    sys.stderr.write(' '.join(parts) + '\n')


def file_bytes(path):
    # 'open' is not avaiable yet.
    with FileIO(path, 'r') as f:
        return f.read()


def iter_keys(prefix):
    l = len(prefix)
    for idx in range(__resource.count()):
        key = __resource.key_by_index(idx)
        if key.startswith(prefix):
            yield key, key[l:]


def iter_py_modules(with_keys=False):
    for key, path in iter_keys(b'resfs/file/' + py_prefix):
        if path.endswith(b'.py'):  # It may also end with '.pyc'.
            mod = _s(path[:-3].replace(b'/', b'.'))
            if with_keys:
                yield key, mod
            else:
                yield mod


def py_src_key(filename):
    return py_prefix + _b(filename)


def iter_prefixes(s):
    i = s.find('.')
    while i >= 0:
        yield s[:i]
        i = s.find('.', i + 1)


def resfs_resolve(path):
    """
    Return the absolute path of a root-relative path if it exists.
    """
    path = _b(path)
    if Y_PYTHON_SOURCE_ROOT:
        if not path.startswith(Y_PYTHON_SOURCE_ROOT):
            path = _b(path_sep).join((Y_PYTHON_SOURCE_ROOT, path))
        if _path_isfile(path):
            return path


def resfs_src(key, resfs_file=False):
    """
    Return the root-relative file path of a resource key.
    """
    if resfs_file:
        key = b'resfs/file/' + _b(key)
    return __resource.find(b'resfs/src/' + _b(key))


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
        return __resource.find(b'resfs/file/' + _b(path))


def resfs_files(prefix=b''):
    """
    List builtin resource file paths.
    """
    return [key[11:] for key, _ in iter_keys(b'resfs/file/' + _b(prefix))]


def mod_path(mod):
    """
    Return the resfs path to the source code of the module with the given name.
    """
    return py_prefix + _b(mod).replace(b'.', b'/') + b'.py'


class ResourceImporter:

    """ A meta_path importer that loads code from built-in resources.
    """

    def __init__(self):
        self.memory = set(iter_py_modules())  # Set of importable module names.
        self.source_map = {}                  # Map from file names to module names.
        self._source_name = {}                # Map from original to altered module names.
        self._package_prefix = ''

        self._before_import_callback = None
        self._after_import_callback = None

        if Y_PYTHON_SOURCE_ROOT and Y_PYTHON_EXTENDED_SOURCE_SEARCH:
            self.arcadia_source_finder = ArcadiaSourceFinder(_s(Y_PYTHON_SOURCE_ROOT))
        else:
            self.arcadia_source_finder = None

        for p in list(self.memory) + list(sys.builtin_module_names):
            for pp in iter_prefixes(p):
                k = pp + '.__init__'
                if k not in self.memory:
                    self.memory.add(k)

    def set_callbacks(self, before_import=None, after_import=None):
        """Callable[[module], None]"""
        self._before_import_callback= before_import
        self._after_import_callback = after_import

    def for_package(self, name):
        import copy
        importer = copy.copy(self)
        importer._package_prefix = name + '.'
        return importer

    def _find_mod_path(self, fullname):
        """Find arcadia relative path by module name"""
        relpath = resfs_src(mod_path(fullname), resfs_file=True)
        if relpath or not self.arcadia_source_finder:
            return relpath
        return self.arcadia_source_finder.get_module_path(fullname)

    def find_spec(self, fullname, path=None, target=None):
        # Поддежка переопределения стандартного distutils из пакетом из setuptools
        if fullname.startswith("distutils."):
            setuptools_path = f"{path_sep}setuptools{path_sep}_distutils"
            if path and len(path) > 0 and setuptools_path in path[0]:
                import importlib
                import importlib.abc

                setuptools_name = "setuptools._distutils.{}".format(fullname.removeprefix("distutils."))
                is_package = self.is_package(setuptools_name)
                if is_package:
                    source = self.get_source(f"{setuptools_name}.__init__")
                    relpath = self._find_mod_path(f"{setuptools_name}.__init__")
                else:
                    source = self.get_source(setuptools_name)
                    relpath = self._find_mod_path(setuptools_name)

                class DistutilsLoader(importlib.abc.Loader):
                    def exec_module(self, module):
                        code = compile(source, _s(relpath), 'exec', dont_inherit=True)
                        module.__file__ = code.co_filename
                        if is_package:
                            module.__path__= [executable + path_sep + setuptools_name.replace('.', path_sep)]

                        _call_with_frames_removed(exec, code, module.__dict__)

                return spec_from_loader(fullname, DistutilsLoader(), is_package=is_package)

        try:
            is_package = self.is_package(fullname)
        except ImportError:
            return None
        return spec_from_loader(fullname, self, is_package=is_package)

    def find_module(self, fullname, path=None):
        """For backward compatibility."""
        spec = self.find_spec(fullname, path)
        return spec.loader if spec is not None else None

    def create_module(self, spec):
        """Use default semantics for module creation."""

    def exec_module(self, module):
        code = self.get_code(module.__name__)
        module.__file__ = code.co_filename
        if self.is_package(module.__name__):
            module.__path__= [executable + path_sep + module.__name__.replace('.', path_sep)]
        # exec(code, module.__dict__)

        # __name__ and __file__ could be overwritten after execution
        # So these two things are needed if wee want to be consistent at some point
        initial_modname = module.__name__
        initial_filename = module.__file__

        if self._before_import_callback:
            self._before_import_callback(initial_modname, initial_filename)

        # “Zero-cost” exceptions are implemented.
        # The cost of try statements is almost eliminated when no exception is raised
        try:
            _call_with_frames_removed(exec, code, module.__dict__)
        finally:
            if self._after_import_callback:
                self._after_import_callback(initial_modname, initial_filename)

    # PEP-302 extension 1 of 3: data loader.
    def get_data(self, path):
        path = _b(path)
        abspath = resfs_resolve(path)
        if abspath:
            return file_bytes(abspath)
        path = path.replace(_b('\\'), _b('/'))
        data = resfs_read(path, builtin=True)
        if data is None:
            raise OSError(path)  # Y_PYTHON_ENTRY_POINT=:resource_files
        return data

    # PEP-302 extension 2 of 3: get __file__ without importing.
    def get_filename(self, fullname):
        modname = fullname
        if self.is_package(fullname):
            fullname += '.__init__'
        relpath = self._find_mod_path(fullname)
        if isinstance(relpath, bytes):
            relpath = _s(relpath)
        return relpath or modname

    # PEP-302 extension 3 of 3: packaging introspection.
    # Used by `linecache` (while printing tracebacks) unless module filename
    # exists on the filesystem.
    def get_source(self, fullname):
        fullname = self._source_name.get(fullname) or fullname
        if self.is_package(fullname):
            fullname += '.__init__'

        relpath = self.get_filename(fullname)
        if relpath:
            abspath = resfs_resolve(relpath)
            if abspath:
                return _s(file_bytes(abspath))
        data = resfs_read(mod_path(fullname))
        return _s(data) if data else ''

    def get_code(self, fullname):
        modname = fullname
        if self.is_package(fullname):
            fullname += '.__init__'

        path = mod_path(fullname)
        relpath = self._find_mod_path(fullname)
        if relpath:
            abspath = resfs_resolve(relpath)
            if abspath:
                data = file_bytes(abspath)
                return compile(data, _s(abspath), 'exec', dont_inherit=True)

        yapyc_path = path + b'.yapyc3'
        yapyc_data = resfs_read(yapyc_path, builtin=True)
        if yapyc_data:
            return marshal.loads(yapyc_data)
        else:
            py_data = resfs_read(path, builtin=True)
            if py_data:
                return compile(py_data, _s(relpath), 'exec', dont_inherit=True)
            else:
                # This covers packages with no __init__.py in resources.
                return compile('', modname, 'exec', dont_inherit=True)

    def is_package(self, fullname):
        if fullname in self.memory:
            return False

        if fullname + '.__init__' in self.memory:
            return True

        if self.arcadia_source_finder:
            return self.arcadia_source_finder.is_package(fullname)

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
            return b'resfs/file/' + _b(filename)

        return b''

    # Extension for pkgutil.iter_modules.
    def iter_modules(self, prefix=''):
        import re
        rx = re.compile(re.escape(self._package_prefix) + r'([^.]+)(\.__init__)?$')
        for p in self.memory:
            m = rx.match(p)
            if m:
                yield prefix + m.group(1), m.group(2) is not None
        if self.arcadia_source_finder:
            for m in self.arcadia_source_finder.iter_modules(self._package_prefix, prefix):
                yield m

    def get_resource_reader(self, fullname):
        import os
        path = os.path.dirname(self.get_filename(fullname))
        return _ResfsResourceReader(self, path)


class _ResfsResourceReader:

    def __init__(self, importer, path):
        self.importer = importer
        self.path = path

    def open_resource(self, resource):
        path = f'{self.path}/{resource}'
        from io import BytesIO
        try:
            return BytesIO(self.importer.get_data(path))
        except OSError:
            raise FileNotFoundError(path)

    def resource_path(self, resource):
        # All resources are in the binary file, so there is no path to the file.
        # Raising FileNotFoundError tells the higher level API to extract the
        # binary data and create a temporary file.
        raise FileNotFoundError

    def is_resource(self, name):
        path = f'{self.path}/{name}'
        try:
            self.importer.get_data(path)
        except OSError:
            return False
        return True

    def contents(self):
        subdirs_seen = set()
        len_path = len(self.path) + 1  # path + /
        for key in resfs_files(f"{self.path}/"):
            relative = key[len_path:]
            res_or_subdir, *other = relative.split(b'/')
            if not other:
                yield _s(res_or_subdir)
            elif res_or_subdir not in subdirs_seen:
                subdirs_seen.add(res_or_subdir)
                yield _s(res_or_subdir)

    def files(self):
        import sitecustomize
        return sitecustomize.ArcadiaResourceContainer(f"resfs/file/{self.path}/")


class BuiltinSubmoduleImporter(BuiltinImporter):
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if path is not None:
            return super().find_spec(fullname, None, target)
        else:
            return None


class ArcadiaSourceFinder:
    """
        Search modules and packages in arcadia source tree.
        See https://wiki.yandex-team.ru/devtools/extended-python-source-search/ for details
    """
    NAMESPACE_PREFIX = b'py/namespace/'
    PY_EXT = '.py'
    YA_MAKE = 'ya.make'
    S_IFDIR = 0o040000

    def __init__(self, source_root):
        self.source_root = source_root
        self.module_path_cache = {'': set()}
        for key, dirty_path in iter_keys(self.NAMESPACE_PREFIX):
            # dirty_path contains unique prefix to prevent repeatable keys in the resource storage
            path = dirty_path.split(b'/', 1)[1]
            namespaces = __resource.find(key).split(b':')
            for n in namespaces:
                package_name = _s(n.rstrip(b'.'))
                self.module_path_cache.setdefault(package_name, set()).add(_s(path))
                # Fill parents with default empty path set if parent doesn't exist in the cache yet
                while package_name:
                    package_name = package_name.rpartition('.')[0]
                    if package_name in self.module_path_cache:
                        break
                    self.module_path_cache.setdefault(package_name, set())
        for package_name in self.module_path_cache.keys():
            self._add_parent_dirs(package_name, visited=set())

    def get_module_path(self, fullname):
        """
            Find file path for module 'fullname'.
            For packages caller pass fullname as 'package.__init__'.
            Return None if nothing is found.
        """
        try:
            if not self.is_package(fullname):
                return _b(self._cache_module_path(fullname))
        except ImportError:
            pass

    def is_package(self, fullname):
        """Check if fullname is a package. Raise ImportError if fullname is not found"""
        path = self._cache_module_path(fullname)
        if isinstance(path, set):
            return True
        if isinstance(path, str):
            return False
        raise ImportError(fullname)

    def iter_modules(self, package_prefix, prefix):
        paths = self._cache_module_path(package_prefix.rstrip('.'))
        if paths is not None:
            # Note: it's ok to yield duplicates because pkgutil discards them

            # Yield from cache
            import re
            rx = re.compile(re.escape(package_prefix) + r'([^.]+)$')
            # Save result to temporary list to prevent 'RuntimeError: dictionary changed size during iteration'
            found = []
            for mod, path in self.module_path_cache.items():
                if path is not None:
                    m = rx.match(mod)
                    if m:
                        found.append((prefix + m.group(1), self.is_package(mod)))
            yield from found

            # Yield from file system
            for path in paths:
                abs_path = _path_join(self.source_root, path)
                for dir_item in _os.listdir(abs_path):
                    if self._path_is_simple_dir(_path_join(abs_path, dir_item)):
                        yield prefix  +  dir_item, True
                    elif dir_item.endswith(self.PY_EXT) and _path_isfile(_path_join(abs_path, dir_item)):
                        yield prefix + dir_item[:-len(self.PY_EXT)], False

    def _isdir(self, path):
        """ Unlike _path_isdir() this function don't follow symlink """
        try:
            stat_info = _os.lstat(path)
        except OSError:
            return False
        return (stat_info.st_mode & 0o170000) == self.S_IFDIR

    def _path_is_simple_dir(self, abs_path):
        """
            Check if path is a directory but doesn't contain ya.make file.
            We don't want to steal directory from nested project and treat it as a package
        """
        return self._isdir(abs_path) and not _path_isfile(_path_join(abs_path, self.YA_MAKE))

    def _find_module_in_paths(self, find_package_only, paths, module):
        """Auxiliary method. See _cache_module_path() for details"""
        if paths:
            package_paths = set()
            for path in paths:
                rel_path = _path_join(path, module)
                if not find_package_only:
                    # Check if file_path is a module
                    module_path = rel_path + self.PY_EXT
                    if _path_isfile(_path_join(self.source_root, module_path)):
                        return module_path
                # Check if file_path is a package
                if self._path_is_simple_dir(_path_join(self.source_root, rel_path)):
                    package_paths.add(rel_path)
            if package_paths:
                return package_paths

    def _cache_module_path(self, fullname, find_package_only=False):
        """
            Find module path or package directory paths and save result in the cache

            find_package_only=True - don't try to find module

            Returns:
               List of relative package paths - for a package
               Relative module path - for a module
               None - module or package is not found
        """
        if fullname not in self.module_path_cache:
            parent, _, tail = fullname.rpartition('.')
            parent_paths = self._cache_module_path(parent, find_package_only=True)
            self.module_path_cache[fullname] = self._find_module_in_paths(find_package_only, parent_paths, tail)
        return self.module_path_cache[fullname]

    def _add_parent_dirs(self, package_name, visited):
        if not package_name or package_name in visited:
            return
        visited.add(package_name)

        parent, _, tail = package_name.rpartition('.')
        self._add_parent_dirs(parent, visited)

        paths = self.module_path_cache[package_name]
        for parent_path in self.module_path_cache[parent]:
            rel_path = _path_join(parent_path, tail)
            if self._path_is_simple_dir(_path_join(self.source_root, rel_path)):
                paths.add(rel_path)


def excepthook(*args, **kws):
    # traceback module cannot be imported at module level, because interpreter
    # is not fully initialized yet

    import traceback

    return traceback.print_exception(*args, **kws)


importer = ResourceImporter()


def executable_path_hook(path):
    if path == executable:
        return importer

    if path.startswith(executable + path_sep):
        return importer.for_package(path[len(executable + path_sep):].replace(path_sep, '.'))

    raise ImportError(path)


def get_path0():
    """
    An incomplete and simplified version of _PyPathConfig_ComputeSysPath0.
    We need this to somewhat properly emulate the behaviour of a normal python interpreter
    when using ya ide venv.

    """
    if not sys.argv:
        return
    argv0 = sys.argv[0]

    have_module_arg = argv0 == '-m'

    if have_module_arg:
        return _os.getcwd()


if YA_IDE_VENV:
    sys.meta_path.append(importer)
    sys.meta_path.append(BuiltinSubmoduleImporter)
    if executable not in sys.path:
        sys.path.append(executable)
    path0 = get_path0()
    if path0 is not None:
        sys.path.insert(0, path0)

    sys.path_hooks.append(executable_path_hook)
else:
    sys.meta_path.insert(0, BuiltinSubmoduleImporter)
    sys.meta_path.insert(0, importer)
    if executable not in sys.path:
        sys.path.insert(0, executable)
    sys.path_hooks.insert(0, executable_path_hook)

sys.path_importer_cache[executable] = importer

# Indicator that modules and resources are built-in rather than on the file system.
sys.is_standalone_binary = True
sys.frozen = True

# Set of names of importable modules.
sys.extra_modules = importer.memory

# Use custom implementation of traceback printer.
# Built-in printer (PyTraceBack_Print) does not support custom module loaders
sys.excepthook = excepthook
