"""
To ensure compatibility from Python ``2.7`` - ``3.x``, a module has been
created. Clearly there is huge need to use conforming syntax.
"""
import errno
import sys
import os
import re
import pkgutil
import warnings
import inspect
import subprocess
try:
    import importlib
except ImportError:
    pass

is_py3 = sys.version_info[0] >= 3
is_py35 = is_py3 and sys.version_info[1] >= 5
py_version = int(str(sys.version_info[0]) + str(sys.version_info[1]))


class DummyFile(object):
    def __init__(self, loader, string):
        self.loader = loader
        self.string = string

    def read(self):
        return self.loader.get_source(self.string)

    def close(self):
        del self.loader


def find_module_py34(string, path=None, full_name=None, is_global_search=True):
    spec = None
    loader = None

    for finder in sys.meta_path:
        if is_global_search and finder != importlib.machinery.PathFinder:
            p = None
        else:
            p = path
        try:
            find_spec = finder.find_spec
        except AttributeError:
            # These are old-school clases that still have a different API, just
            # ignore those.
            continue

        spec = find_spec(string, p)
        if spec is not None:
            loader = spec.loader
            if loader is None and not spec.has_location:
                # This is a namespace package.
                full_name = string if not path else full_name
                implicit_ns_info = ImplicitNSInfo(full_name, spec.submodule_search_locations._path)
                return None, implicit_ns_info, False
            break

    return find_module_py33(string, path, loader)


def find_module_py33(string, path=None, loader=None, full_name=None, is_global_search=True):
    loader = loader or importlib.machinery.PathFinder.find_module(string, path)

    if loader is None and path is None:  # Fallback to find builtins
        try:
            with warnings.catch_warnings(record=True):
                # Mute "DeprecationWarning: Use importlib.util.find_spec()
                # instead." While we should replace that in the future, it's
                # probably good to wait until we deprecate Python 3.3, since
                # it was added in Python 3.4 and find_loader hasn't been
                # removed in 3.6.
                loader = importlib.find_loader(string)
        except ValueError as e:
            # See #491. Importlib might raise a ValueError, to avoid this, we
            # just raise an ImportError to fix the issue.
            raise ImportError("Originally  " + repr(e))

    if loader is None:
        raise ImportError("Couldn't find a loader for {}".format(string))

    try:
        is_package = loader.is_package(string)
        if is_package:
            if hasattr(loader, 'path'):
                module_path = os.path.dirname(loader.path)
            else:
                # At least zipimporter does not have path attribute
                module_path = os.path.dirname(loader.get_filename(string))
            if hasattr(loader, 'archive'):
                module_file = DummyFile(loader, string)
            else:
                module_file = None
        else:
            module_path = loader.get_filename(string)
            module_file = DummyFile(loader, string)
    except AttributeError:
        # ExtensionLoader has not attribute get_filename, instead it has a
        # path attribute that we can use to retrieve the module path
        try:
            module_path = loader.path
            module_file = DummyFile(loader, string)
        except AttributeError:
            module_path = string
            module_file = None
        finally:
            is_package = False

    if hasattr(loader, 'archive'):
        module_path = loader.archive

    return module_file, module_path, is_package


def find_module_pre_py34(string, path=None, full_name=None, is_global_search=True):
    # This import is here, because in other places it will raise a
    # DeprecationWarning.
    import imp
    try:
        module_file, module_path, description = imp.find_module(string, path)
        module_type = description[2]
        return module_file, module_path, module_type is imp.PKG_DIRECTORY
    except ImportError:
        pass

    if path is None:
        path = sys.path
    for item in path:
        loader = pkgutil.get_importer(item)
        if loader:
            try:
                loader = loader.find_module(string)
                if loader:
                    is_package = loader.is_package(string)
                    is_archive = hasattr(loader, 'archive')
                    module_path = loader.get_filename(string)
                    if is_package:
                        module_path = os.path.dirname(module_path)
                    if is_archive:
                        module_path = loader.archive
                    file = None
                    if not is_package or is_archive:
                        file = DummyFile(loader, string)
                    return file, module_path, is_package
            except ImportError:
                pass
    raise ImportError("No module named {}".format(string))


find_module = find_module_py34 if is_py3 else find_module_pre_py34
find_module.__doc__ = """
Provides information about a module.

This function isolates the differences in importing libraries introduced with
python 3.3 on; it gets a module name and optionally a path. It will return a
tuple containin an open file for the module (if not builtin), the filename
or the name of the module if it is a builtin one and a boolean indicating
if the module is contained in a package.
"""


def _iter_modules(paths, prefix=''):
    # Copy of pkgutil.iter_modules adapted to work with namespaces

    for path in paths:
        importer = pkgutil.get_importer(path)

        if not isinstance(importer, importlib.machinery.FileFinder):
            # We're only modifying the case for FileFinder. All the other cases
            # still need to be checked (like zip-importing). Do this by just
            # calling the pkgutil version.
            for mod_info in pkgutil.iter_modules([path], prefix):
                yield mod_info
            continue

        # START COPY OF pkutils._iter_file_finder_modules.
        if importer.path is None or not os.path.isdir(importer.path):
            return

        yielded = {}

        try:
            filenames = os.listdir(importer.path)
        except OSError:
            # ignore unreadable directories like import does
            filenames = []
        filenames.sort()  # handle packages before same-named modules

        for fn in filenames:
            modname = inspect.getmodulename(fn)
            if modname == '__init__' or modname in yielded:
                continue

            # jedi addition: Avoid traversing special directories
            if fn.startswith('.') or fn == '__pycache__':
                continue

            path = os.path.join(importer.path, fn)
            ispkg = False

            if not modname and os.path.isdir(path) and '.' not in fn:
                modname = fn
                # A few jedi modifications: Don't check if there's an
                # __init__.py
                try:
                    os.listdir(path)
                except OSError:
                    # ignore unreadable directories like import does
                    continue
                ispkg = True

            if modname and '.' not in modname:
                yielded[modname] = 1
                yield importer, prefix + modname, ispkg
        # END COPY


iter_modules = _iter_modules if py_version >= 34 else pkgutil.iter_modules


class ImplicitNSInfo(object):
    """Stores information returned from an implicit namespace spec"""
    def __init__(self, name, paths):
        self.name = name
        self.paths = paths


if is_py3:
    all_suffixes = importlib.machinery.all_suffixes
else:
    def all_suffixes():
        # Is deprecated and raises a warning in Python 3.6.
        import imp
        return [suffix for suffix, _, _ in imp.get_suffixes()]


# unicode function
try:
    unicode = unicode
except NameError:
    unicode = str


# re-raise function
if is_py3:
    def reraise(exception, traceback):
        raise exception.with_traceback(traceback)
else:
    eval(compile("""
def reraise(exception, traceback):
    raise exception, None, traceback
""", 'blub', 'exec'))

reraise.__doc__ = """
Re-raise `exception` with a `traceback` object.

Usage::

    reraise(Exception, sys.exc_info()[2])

"""


class Python3Method(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype):
        if obj is None:
            return lambda *args, **kwargs: self.func(*args, **kwargs)
        else:
            return lambda *args, **kwargs: self.func(obj, *args, **kwargs)


def use_metaclass(meta, *bases):
    """ Create a class with a metaclass. """
    if not bases:
        bases = (object,)
    return meta("Py2CompatibilityMetaClass", bases, {})


try:
    encoding = sys.stdout.encoding
    if encoding is None:
        encoding = 'utf-8'
except AttributeError:
    encoding = 'ascii'


def u(string, errors='strict'):
    """Cast to unicode DAMMIT!
    Written because Python2 repr always implicitly casts to a string, so we
    have to cast back to a unicode (and we now that we always deal with valid
    unicode, because we check that in the beginning).
    """
    if isinstance(string, bytes):
        return unicode(string, encoding='UTF-8', errors=errors)
    return string


def cast_path(obj):
    """
    Take a bytes or str path and cast it to unicode.

    Apparently it is perfectly fine to pass both byte and unicode objects into
    the sys.path. This probably means that byte paths are normal at other
    places as well.

    Since this just really complicates everything and Python 2.7 will be EOL
    soon anyway, just go with always strings.
    """
    return u(obj, errors='replace')


def force_unicode(obj):
    # Intentionally don't mix those two up, because those two code paths might
    # be different in the future (maybe windows?).
    return cast_path(obj)


try:
    import builtins  # module name in python 3
except ImportError:
    import __builtin__ as builtins  # noqa: F401


import ast  # noqa: F401


def literal_eval(string):
    return ast.literal_eval(string)


try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest  # Python 2  # noqa: F401

try:
    FileNotFoundError = FileNotFoundError
except NameError:
    FileNotFoundError = IOError

try:
    NotADirectoryError = NotADirectoryError
except NameError:
    NotADirectoryError = IOError

try:
    PermissionError = PermissionError
except NameError:
    PermissionError = IOError


def no_unicode_pprint(dct):
    """
    Python 2/3 dict __repr__ may be different, because of unicode differens
    (with or without a `u` prefix). Normally in doctests we could use `pprint`
    to sort dicts and check for equality, but here we have to write a separate
    function to do that.
    """
    import pprint
    s = pprint.pformat(dct)
    print(re.sub("u'", "'", s))


def print_to_stderr(*args):
    if is_py3:
        eval("print(*args, file=sys.stderr)")
    else:
        print >> sys.stderr, args
    sys.stderr.flush()


def utf8_repr(func):
    """
    ``__repr__`` methods in Python 2 don't allow unicode objects to be
    returned. Therefore cast them to utf-8 bytes in this decorator.
    """
    def wrapper(self):
        result = func(self)
        if isinstance(result, unicode):
            return result.encode('utf-8')
        else:
            return result

    if is_py3:
        return func
    else:
        return wrapper


if is_py3:
    import queue
else:
    import Queue as queue  # noqa: F401

try:
    # Attempt to load the C implementation of pickle on Python 2 as it is way
    # faster.
    import cPickle as pickle
except ImportError:
    import pickle
if sys.version_info[:2] == (3, 3):
    """
    Monkeypatch the unpickler in Python 3.3. This is needed, because the
    argument `encoding='bytes'` is not supported in 3.3, but badly needed to
    communicate with Python 2.
    """

    class NewUnpickler(pickle._Unpickler):
        dispatch = dict(pickle._Unpickler.dispatch)

        def _decode_string(self, value):
            # Used to allow strings from Python 2 to be decoded either as
            # bytes or Unicode strings.  This should be used only with the
            # STRING, BINSTRING and SHORT_BINSTRING opcodes.
            if self.encoding == "bytes":
                return value
            else:
                return value.decode(self.encoding, self.errors)

        def load_string(self):
            data = self.readline()[:-1]
            # Strip outermost quotes
            if len(data) >= 2 and data[0] == data[-1] and data[0] in b'"\'':
                data = data[1:-1]
            else:
                raise pickle.UnpicklingError("the STRING opcode argument must be quoted")
            self.append(self._decode_string(pickle.codecs.escape_decode(data)[0]))
        dispatch[pickle.STRING[0]] = load_string

        def load_binstring(self):
            # Deprecated BINSTRING uses signed 32-bit length
            len, = pickle.struct.unpack('<i', self.read(4))
            if len < 0:
                raise pickle.UnpicklingError("BINSTRING pickle has negative byte count")
            data = self.read(len)
            self.append(self._decode_string(data))
        dispatch[pickle.BINSTRING[0]] = load_binstring

        def load_short_binstring(self):
            len = self.read(1)[0]
            data = self.read(len)
            self.append(self._decode_string(data))
        dispatch[pickle.SHORT_BINSTRING[0]] = load_short_binstring

    def load(file, fix_imports=True, encoding="ASCII", errors="strict"):
        return NewUnpickler(file, fix_imports=fix_imports,
                            encoding=encoding, errors=errors).load()

    def loads(s, fix_imports=True, encoding="ASCII", errors="strict"):
        if isinstance(s, str):
            raise TypeError("Can't load pickle from unicode string")
        file = pickle.io.BytesIO(s)
        return NewUnpickler(file, fix_imports=fix_imports,
                            encoding=encoding, errors=errors).load()

    pickle.Unpickler = NewUnpickler
    pickle.load = load
    pickle.loads = loads


def pickle_load(file):
    try:
        if is_py3:
            return pickle.load(file, encoding='bytes')
        return pickle.load(file)
    # Python on Windows don't throw EOF errors for pipes. So reraise them with
    # the correct type, which is caught upwards.
    except OSError:
        if sys.platform == 'win32':
            raise EOFError()
        raise


def pickle_dump(data, file, protocol):
    try:
        pickle.dump(data, file, protocol)
        # On Python 3.3 flush throws sometimes an error even though the writing
        # operation should be completed.
        file.flush()
    # Python on Windows don't throw EPIPE errors for pipes. So reraise them with
    # the correct type and error number.
    except OSError:
        if sys.platform == 'win32':
            raise IOError(errno.EPIPE, "Broken pipe")
        raise


# Determine the highest protocol version compatible for a given list of Python
# versions.
def highest_pickle_protocol(python_versions):
    protocol = 4
    for version in python_versions:
        if version[0] == 2:
            # The minimum protocol version for the versions of Python that we
            # support (2.7 and 3.3+) is 2.
            return 2
        if version[1] < 4:
            protocol = 3
    return protocol


try:
    from inspect import Parameter
except ImportError:
    class Parameter(object):
        POSITIONAL_ONLY = object()
        POSITIONAL_OR_KEYWORD = object()
        VAR_POSITIONAL = object()
        KEYWORD_ONLY = object()
        VAR_KEYWORD = object()


class GeneralizedPopen(subprocess.Popen):
    def __init__(self, *args, **kwargs):
        if os.name == 'nt':
            try:
                # Was introduced in Python 3.7.
                CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
            except AttributeError:
                CREATE_NO_WINDOW = 0x08000000
            kwargs['creationflags'] = CREATE_NO_WINDOW
        # The child process doesn't need file descriptors except 0, 1, 2.
        # This is unix only.
        kwargs['close_fds'] = 'posix' in sys.builtin_module_names
        super(GeneralizedPopen, self).__init__(*args, **kwargs)


# shutil.which is not available on Python 2.7.
def which(cmd, mode=os.F_OK | os.X_OK, path=None):
    """Given a command, mode, and a PATH string, return the path which
    conforms to the given mode on the PATH, or None if there is no such
    file.

    `mode` defaults to os.F_OK | os.X_OK. `path` defaults to the result
    of os.environ.get("PATH"), or can be overridden with a custom search
    path.

    """
    # Check that a given file can be accessed with the correct mode.
    # Additionally check that `file` is not a directory, as on Windows
    # directories pass the os.access check.
    def _access_check(fn, mode):
        return (os.path.exists(fn) and os.access(fn, mode)
                and not os.path.isdir(fn))

    # If we're given a path with a directory part, look it up directly rather
    # than referring to PATH directories. This includes checking relative to the
    # current directory, e.g. ./script
    if os.path.dirname(cmd):
        if _access_check(cmd, mode):
            return cmd
        return None

    if path is None:
        path = os.environ.get("PATH", os.defpath)
    if not path:
        return None
    path = path.split(os.pathsep)

    if sys.platform == "win32":
        # The current directory takes precedence on Windows.
        if os.curdir not in path:
            path.insert(0, os.curdir)

        # PATHEXT is necessary to check on Windows.
        pathext = os.environ.get("PATHEXT", "").split(os.pathsep)
        # See if the given file matches any of the expected path extensions.
        # This will allow us to short circuit when given "python.exe".
        # If it does match, only test that one, otherwise we have to try
        # others.
        if any(cmd.lower().endswith(ext.lower()) for ext in pathext):
            files = [cmd]
        else:
            files = [cmd + ext for ext in pathext]
    else:
        # On other platforms you don't have things like PATHEXT to tell you
        # what file suffixes are executable, so just pass on cmd as-is.
        files = [cmd]

    seen = set()
    for dir in path:
        normdir = os.path.normcase(dir)
        if normdir not in seen:
            seen.add(normdir)
            for thefile in files:
                name = os.path.join(dir, thefile)
                if _access_check(name, mode):
                    return name
    return None
