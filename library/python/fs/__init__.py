# coding: utf-8

import codecs
import errno
import logging
import os
import random
import shutil
import six
import stat
import sys

import library.python.func
import library.python.strings
import library.python.windows

logger = logging.getLogger(__name__)


try:
    WindowsError
except NameError:
    WindowsError = None


_diehard_win_tries = 10
errorfix_win = library.python.windows.errorfix


class CustomFsError(OSError):
    def __init__(self, errno, message='', filename=None):
        super(CustomFsError, self).__init__(message)
        self.errno = errno
        self.strerror = os.strerror(errno)
        self.filename = filename


# Directories creation
# If dst is already exists and is a directory - does nothing
# Throws OSError
@errorfix_win
def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST or not os.path.isdir(path):
            raise


# Directories creation
# If dst is already exists and is a directory - does nothing
# Returns path
# Throws OSError
@errorfix_win
def create_dirs(path):
    ensure_dir(path)
    return path


# Atomic file/directory move (rename)
# Doesn't guarantee dst replacement
# Atomic if no device boundaries are crossed
# Depends on ctypes on Windows
# Throws OSError
# On Unix, if dst exists:
#   if dst is file or empty dir - replaces it
#   if src is dir and dst is not dir - throws OSError (errno ENOTDIR)
#   if src is dir and dst is non-empty dir - throws OSError (errno ENOTEMPTY)
#   if src is file and dst is dir - throws OSError (errno EISDIR)
# On Windows, if dst exists - throws OSError (errno EEXIST)
@errorfix_win
@library.python.windows.diehard(library.python.windows.RETRIABLE_FILE_ERRORS, tries=_diehard_win_tries)
def move(src, dst):
    os.rename(src, dst)


# Atomic replacing file move (rename)
# Replaces dst if exists and not a dir
# Doesn't guarantee dst dir replacement
# Atomic if no device boundaries are crossed
# Depends on ctypes on Windows
# Throws OSError
# On Unix, if dst exists:
#   if dst is file - replaces it
#   if dst is dir - throws OSError (errno EISDIR)
# On Windows, if dst exists:
#   if dst is file - replaces it
#   if dst is dir - throws OSError (errno EACCES)
@errorfix_win
@library.python.windows.diehard(library.python.windows.RETRIABLE_FILE_ERRORS, tries=_diehard_win_tries)
def replace_file(src, dst):
    if library.python.windows.on_win():
        library.python.windows.replace_file(src, dst)
    else:
        os.rename(src, dst)


# File/directory replacing move (rename)
# Removes dst if exists
# Non-atomic
# Depends on ctypes on Windows
# Throws OSError
@errorfix_win
def replace(src, dst):
    try:
        move(src, dst)
    except OSError as e:
        if e.errno not in (errno.EEXIST, errno.EISDIR, errno.ENOTDIR, errno.ENOTEMPTY):
            raise
        remove_tree(dst)
        move(src, dst)


# Atomic file remove
# Throws OSError
@errorfix_win
@library.python.windows.diehard(library.python.windows.RETRIABLE_FILE_ERRORS, tries=_diehard_win_tries)
def remove_file(path):
    os.remove(path)


# Atomic empty directory remove
# Throws OSError
@errorfix_win
@library.python.windows.diehard(library.python.windows.RETRIABLE_DIR_ERRORS, tries=_diehard_win_tries)
def remove_dir(path):
    os.rmdir(path)


def fix_path_encoding(path):
    return library.python.strings.to_str(path, library.python.strings.fs_encoding())


# File/directory remove
# Non-atomic
# Throws OSError, AssertionError
@errorfix_win
def remove_tree(path):
    @library.python.windows.diehard(library.python.windows.RETRIABLE_DIR_ERRORS, tries=_diehard_win_tries)
    def rmtree(path):
        if library.python.windows.on_win():
            library.python.windows.rmtree(path)
        else:
            shutil.rmtree(fix_path_encoding(path))

    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or stat.S_ISREG(st.st_mode):
        remove_file(path)
    elif stat.S_ISDIR(st.st_mode):
        rmtree(path)
    else:
        assert False


# File/directory remove ignoring errors
# Non-atomic
@errorfix_win
def remove_tree_safe(path):
    try:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode) or stat.S_ISREG(st.st_mode):
            os.remove(path)
        elif stat.S_ISDIR(st.st_mode):
            shutil.rmtree(fix_path_encoding(path), ignore_errors=True)
    # XXX
    except UnicodeDecodeError as e:
        logging.exception(u'remove_tree_safe with argument %s raise exception: %s', path, e)
        raise
    except OSError:
        pass


# File/directory remove
# If path doesn't exist - does nothing
# Non-atomic
# Throws OSError, AssertionError
@errorfix_win
def ensure_removed(path):
    try:
        remove_tree(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


# Atomic file hardlink
# Dst must not exist
# Depends on ctypes on Windows
# Throws OSError
# If dst exists - throws OSError (errno EEXIST)
@errorfix_win
def hardlink(src, lnk):
    if library.python.windows.on_win():
        library.python.windows.hardlink(src, lnk)
    else:
        os.link(src, lnk)


@errorfix_win
def hardlink_or_copy(src, lnk):
    def should_fallback_to_copy(exc):
        if WindowsError is not None and isinstance(exc, WindowsError) and exc.winerror == 1142:  # too many hardlinks
            return True
        # cross-device hardlink or too many hardlinks, or some known WSL error
        if isinstance(exc, OSError) and exc.errno in (
            errno.EXDEV,
            errno.EMLINK,
            errno.EINVAL,
            errno.EACCES,
            errno.EPERM,
        ):
            return True
        return False

    try:
        hardlink(src, lnk)
    except Exception as e:
        logger.debug('Failed to hardlink %s to %s with error %s, will copy it', src, lnk, repr(e))
        if should_fallback_to_copy(e):
            copy2(src, lnk, follow_symlinks=False)
        else:
            raise


# Atomic file/directory symlink (Unix only)
# Dst must not exist
# Throws OSError
# If dst exists - throws OSError (errno EEXIST)
@errorfix_win
def symlink(src, lnk):
    if library.python.windows.on_win():
        library.python.windows.run_disabled(src, lnk)
    else:
        os.symlink(src, lnk)


# shutil.copy2 with follow_symlinks=False parameter (Unix only)
def copy2(src, lnk, follow_symlinks=True):
    if six.PY3:
        shutil.copy2(src, lnk, follow_symlinks=follow_symlinks)
        return

    if follow_symlinks or not os.path.islink(src):
        shutil.copy2(src, lnk)
        return

    symlink(os.readlink(src), lnk)


# Recursively hardlink directory
# Uses plain hardlink for files
# Dst must not exist
# Non-atomic
# Throws OSError
@errorfix_win
def hardlink_tree(src, dst):
    if not os.path.exists(src):
        raise CustomFsError(errno.ENOENT, filename=src)
    if os.path.isfile(src):
        hardlink(src, dst)
        return
    for dirpath, _, filenames in walk_relative(src):
        src_dirpath = os.path.join(src, dirpath) if dirpath != '.' else src
        dst_dirpath = os.path.join(dst, dirpath) if dirpath != '.' else dst
        os.mkdir(dst_dirpath)
        for filename in filenames:
            hardlink(os.path.join(src_dirpath, filename), os.path.join(dst_dirpath, filename))


# File copy
# throws EnvironmentError (OSError, IOError)
@errorfix_win
def copy_file(src, dst, copy_function=shutil.copy2):
    if os.path.isdir(dst):
        raise CustomFsError(errno.EISDIR, filename=dst)
    copy_function(src, dst)


# File/directory copy
# throws EnvironmentError (OSError, IOError, shutil.Error)
@errorfix_win
def copy_tree(src, dst, copy_function=shutil.copy2):
    if os.path.isfile(src):
        copy_file(src, dst, copy_function=copy_function)
        return
    copytree3(src, dst, copy_function=copy_function)


# File read
# Throws OSError
@errorfix_win
def read_file(path, binary=True):
    with open(path, 'r' + ('b' if binary else '')) as f:
        return f.read()


# Text file read
# Throws OSError
@errorfix_win
def read_text(path):
    return read_file(path, binary=False)


# Decoding file read
# Throws OSError
@errorfix_win
def read_file_unicode(path, binary=True, enc='utf-8'):
    if not binary:
        if six.PY2:
            with open(path, 'r') as f:
                return library.python.strings.to_unicode(f.read(), enc)
        else:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
    # codecs.open is always binary
    with codecs.open(path, 'r', encoding=enc, errors=library.python.strings.ENCODING_ERRORS_POLICY) as f:
        return f.read()


@errorfix_win
def open_file(*args, **kwargs):
    return (
        library.python.windows.open_file(*args, **kwargs) if library.python.windows.on_win() else open(*args, **kwargs)
    )


# Atomic file write
# Throws OSError
@errorfix_win
def write_file(path, data, binary=True):
    dir_path = os.path.dirname(path)
    if dir_path:
        ensure_dir(dir_path)
    tmp_path = path + '.tmp.' + str(random.random())
    with open_file(tmp_path, 'w' + ('b' if binary else '')) as f:
        if not isinstance(data, bytes) and binary:
            data = data.encode('UTF-8')
        f.write(data)
    replace_file(tmp_path, path)


# Atomic text file write
# Throws OSError
@errorfix_win
def write_text(path, data):
    write_file(path, data, binary=False)


# File size
# Throws OSError
@errorfix_win
def get_file_size(path):
    return os.path.getsize(path)


# File/directory size
# Non-recursive mode for directory counts size for immediates
# While raise_all_errors is set to False, file size fallbacks to zero in case of getsize errors
# Throws OSError
@errorfix_win
def get_tree_size(path, recursive=False, raise_all_errors=False):
    if os.path.isfile(path):
        return get_file_size(path)
    total_size = 0
    for dir_path, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(dir_path, f)
            try:
                total_size += get_file_size(fp)
            except OSError as e:
                if raise_all_errors:
                    raise
                logger.debug("Cannot calculate file size: %s", e)
        if not recursive:
            break
    return total_size


# Directory copy ported from Python 3
def copytree3(
    src,
    dst,
    symlinks=False,
    ignore=None,
    copy_function=shutil.copy2,
    ignore_dangling_symlinks=False,
    dirs_exist_ok=False,
):
    """Recursively copy a directory tree.

    The copytree3 is a port of shutil.copytree function from python-3.2.
    It has additional useful parameters and may be helpful while we are
    on python-2.x. It has to be removed as soon as we have moved to
    python-3.2 or higher.

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied. If the file pointed by the symlink doesn't
    exist, an exception will be added in the list of errors raised in
    an Error exception at the end of the copy process.

    You can set the optional ignore_dangling_symlinks flag to true if you
    want to silence this exception. Notice that this has no effect on
    platforms that don't support os.symlink.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree3(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree3() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    The optional copy_function argument is a callable that will be used
    to copy each file. It will be called with the source path and the
    destination path as arguments. By default, copy2() is used, but any
    function that supports the same signature (like copy()) can be used.

    """
    names = os.listdir(src)
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    if not (dirs_exist_ok and os.path.isdir(dst)):
        os.makedirs(dst)

    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.islink(srcname):
                linkto = os.readlink(srcname)
                if symlinks:
                    # We can't just leave it to `copy_function` because legacy
                    # code with a custom `copy_function` may rely on copytree3
                    # doing the right thing.
                    os.symlink(linkto, dstname)
                else:
                    # ignore dangling symlink if the flag is on
                    if not os.path.exists(linkto) and ignore_dangling_symlinks:
                        continue
                    # otherwise let the copy occurs. copy2 will raise an error
                    copy_function(srcname, dstname)
            elif os.path.isdir(srcname):
                copytree3(srcname, dstname, symlinks, ignore, copy_function, dirs_exist_ok=dirs_exist_ok)
            else:
                # Will raise a SpecialFileError for unsupported file types
                copy_function(srcname, dstname)
        # catch the Error from the recursive copytree3 so that we can
        # continue with other files
        except shutil.Error as err:
            errors.extend(err.args[0])
        except EnvironmentError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        shutil.copystat(src, dst)
    except OSError as why:
        if WindowsError is not None and isinstance(why, WindowsError):
            # Copying file access times may fail on Windows
            pass
        else:
            errors.extend((src, dst, str(why)))
    if errors:
        raise shutil.Error(errors)


def walk_relative(path, topdown=True, onerror=None, followlinks=False):
    for dirpath, dirnames, filenames in os.walk(path, topdown=topdown, onerror=onerror, followlinks=followlinks):
        yield os.path.relpath(dirpath, path), dirnames, filenames


def supports_clone():
    if 'darwin' in sys.platform:
        import platform

        return list(map(int, platform.mac_ver()[0].split('.'))) >= [10, 13]
    return False


def commonpath(paths):
    assert paths
    if len(paths) == 1:
        return next(iter(paths))

    split_paths = [path.split(os.sep) for path in paths]
    smin = min(split_paths)
    smax = max(split_paths)

    common = smin
    for i, c in enumerate(smin):
        if c != smax[i]:
            common = smin[:i]
            break

    return os.path.sep.join(common)


def set_execute_bits(filename):
    stm = os.stat(filename).st_mode
    exe = stm | 0o111
    if stm != exe:
        os.chmod(filename, exe)
