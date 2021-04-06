# coding=utf-8

import errno
import os
import pytest
import shutil
import six

import library.python.fs
import library.python.strings
import library.python.tmp
import library.python.windows

import yatest.common


def in_env(case):
    def wrapped_case(*args, **kwargs):
        with library.python.tmp.temp_dir() as temp_dir:
            case(lambda path: os.path.join(temp_dir, path))

    return wrapped_case


def mkfile(path, data=''):
    with open(path, 'wb') as f:
        if data:
            f.write(data) if isinstance(data, six.binary_type) else f.write(
                data.encode(library.python.strings.fs_encoding())
            )


def mktree_example(path, name):
    os.mkdir(path(name))
    mkfile(path(name + '/file1'), 'FILE1')
    os.mkdir(path(name + '/dir1'))
    os.mkdir(path(name + '/dir2'))
    mkfile(path(name + '/dir2/file2'), 'FILE2')
    mkfile(path(name + '/dir2/file3'), 'FILE3')


def file_data(path):
    with open(path, 'rb') as f:
        return f.read().decode('utf-8')


def serialize_tree(path):
    if os.path.isfile(path):
        return file_data(path)
    data = {'dirs': set(), 'files': {}}
    for dirpath, dirnames, filenames in os.walk(path):
        dirpath_rel = os.path.relpath(dirpath, path)
        if dirpath_rel == '.':
            dirpath_rel = ''
        data['dirs'].update(set(os.path.join(dirpath_rel, x) for x in dirnames))
        data['files'].update({os.path.join(dirpath_rel, x): file_data(os.path.join(dirpath, x)) for x in filenames})
    return data


def trees_equal(dir1, dir2):
    return serialize_tree(dir1) == serialize_tree(dir2)


def inodes_unsupported():
    return library.python.windows.on_win()


def inodes_equal(path1, path2):
    return os.stat(path1).st_ino == os.stat(path2).st_ino


def gen_error_access_denied():
    if library.python.windows.on_win():
        err = WindowsError()
        err.errno = errno.EACCES
        err.strerror = ''
        err.winerror = library.python.windows.ERRORS['ACCESS_DENIED']
    else:
        err = OSError()
        err.errno = errno.EACCES
        err.strerror = os.strerror(err.errno)
    err.filename = 'unknown/file'
    raise err


def test_errorfix_win():
    @library.python.fs.errorfix_win
    def erroneous_func():
        gen_error_access_denied()

    with pytest.raises(OSError) as errinfo:
        erroneous_func()
    assert errinfo.value.errno == errno.EACCES
    assert errinfo.value.filename == 'unknown/file'
    # See transcode_error, which encodes strerror, in library/python/windows/__init__.py
    assert isinstance(errinfo.value.strerror, (six.binary_type, six.text_type))
    assert errinfo.value.strerror


def test_custom_fs_error():
    with pytest.raises(OSError) as errinfo:
        raise library.python.fs.CustomFsError(errno.EACCES, filename='some/file')
    assert errinfo.value.errno == errno.EACCES
    # See transcode_error, which encodes strerror, in library/python/windows/__init__.py
    assert isinstance(errinfo.value.strerror, (six.binary_type, six.text_type))
    assert errinfo.value.filename == 'some/file'


@in_env
def test_ensure_dir(path):
    library.python.fs.ensure_dir(path('dir/subdir'))
    assert os.path.isdir(path('dir'))
    assert os.path.isdir(path('dir/subdir'))


@in_env
def test_ensure_dir_exists(path):
    os.makedirs(path('dir/subdir'))
    library.python.fs.ensure_dir(path('dir/subdir'))
    assert os.path.isdir(path('dir'))
    assert os.path.isdir(path('dir/subdir'))


@in_env
def test_ensure_dir_exists_partly(path):
    os.mkdir(path('dir'))
    library.python.fs.ensure_dir(path('dir/subdir'))
    assert os.path.isdir(path('dir'))
    assert os.path.isdir(path('dir/subdir'))


@in_env
def test_ensure_dir_exists_file(path):
    mkfile(path('dir'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.ensure_dir(path('dir/subdir'))
    # ENOENT on Windows!
    assert errinfo.value.errno in (errno.ENOTDIR, errno.ENOENT)
    assert os.path.isfile(path('dir'))


@in_env
def test_create_dirs(path):
    assert library.python.fs.create_dirs(path('dir/subdir')) == path('dir/subdir')
    assert os.path.isdir(path('dir'))
    assert os.path.isdir(path('dir/subdir'))


@in_env
def test_move_file(path):
    mkfile(path('src'), 'SRC')
    library.python.fs.move(path('src'), path('dst'))
    assert not os.path.isfile(path('src'))
    assert os.path.isfile(path('dst'))
    assert file_data(path('dst')) == 'SRC'


@in_env
def test_move_file_no_src(path):
    with pytest.raises(OSError) as errinfo:
        library.python.fs.move(path('src'), path('dst'))
    assert errinfo.value.errno == errno.ENOENT


@in_env
def test_move_file_exists(path):
    mkfile(path('src'), 'SRC')
    mkfile(path('dst'), 'DST')
    if library.python.windows.on_win():
        # move is platform-dependent, use replace_file for dst replacement on all platforms
        with pytest.raises(OSError) as errinfo:
            library.python.fs.move(path('src'), path('dst'))
        assert errinfo.value.errno == errno.EEXIST
        assert os.path.isfile(path('src'))
        assert os.path.isfile(path('dst'))
        assert file_data(path('dst')) == 'DST'
    else:
        library.python.fs.move(path('src'), path('dst'))
        assert not os.path.isfile(path('src'))
        assert os.path.isfile(path('dst'))
        assert file_data(path('dst')) == 'SRC'


@in_env
def test_move_file_exists_dir_empty(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.move(path('src'), path('dst'))
    assert errinfo.value.errno in (errno.EEXIST, errno.EISDIR)
    assert os.path.isfile(path('src'))
    assert os.path.isdir(path('dst'))
    assert not os.path.isfile(path('dst/src'))


@in_env
def test_move_file_exists_dir_nonempty(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst'))
    mkfile(path('dst/dst_file'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.move(path('src'), path('dst'))
    assert errinfo.value.errno in (errno.EEXIST, errno.EISDIR)
    assert os.path.isfile(path('src'))
    assert os.path.isdir(path('dst'))
    assert os.path.isfile(path('dst/dst_file'))
    assert not os.path.isfile(path('dst/src'))


@in_env
def test_move_dir(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    library.python.fs.move(path('src'), path('dst'))
    assert not os.path.isdir(path('src'))
    assert os.path.isdir(path('dst'))
    assert os.path.isfile(path('dst/src_file'))


@in_env
def test_move_dir_exists_empty(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    os.mkdir(path('dst'))
    if library.python.windows.on_win():
        # move is platform-dependent, use non-atomic replace for directory replacement
        with pytest.raises(OSError) as errinfo:
            library.python.fs.move(path('src'), path('dst'))
        assert errinfo.value.errno == errno.EEXIST
        assert os.path.isdir(path('src'))
        assert os.path.isdir(path('dst'))
        assert not os.path.isfile(path('dst/src_file'))
    else:
        library.python.fs.move(path('src'), path('dst'))
        assert not os.path.isdir(path('src'))
        assert os.path.isdir(path('dst'))
        assert os.path.isfile(path('dst/src_file'))


@in_env
def test_move_dir_exists_nonempty(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    os.mkdir(path('dst'))
    mkfile(path('dst/dst_file'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.move(path('src'), path('dst'))
    assert errinfo.value.errno in (errno.EEXIST, errno.ENOTEMPTY)
    assert os.path.isdir(path('src'))
    assert os.path.isfile(path('src/src_file'))
    assert os.path.isdir(path('dst'))
    assert not os.path.isfile(path('dst/src_file'))
    assert os.path.isfile(path('dst/dst_file'))


@in_env
def test_move_dir_exists_file(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    mkfile(path('dst'), 'DST')
    with pytest.raises(OSError) as errinfo:
        library.python.fs.move(path('src'), path('dst'))
    assert errinfo.value.errno in (errno.EEXIST, errno.ENOTDIR)
    assert os.path.isdir(path('src'))
    assert os.path.isfile(path('dst'))
    assert file_data(path('dst')) == 'DST'


@in_env
def test_replace_file(path):
    mkfile(path('src'), 'SRC')
    library.python.fs.replace_file(path('src'), path('dst'))
    assert not os.path.isfile(path('src'))
    assert os.path.isfile(path('dst'))
    assert file_data(path('dst')) == 'SRC'

    mkfile(path('src'), 'SRC')
    library.python.fs.replace(path('src'), path('dst2'))
    assert not os.path.isfile(path('src'))
    assert os.path.isfile(path('dst2'))
    assert file_data(path('dst2')) == 'SRC'


@in_env
def test_replace_file_no_src(path):
    with pytest.raises(OSError) as errinfo:
        library.python.fs.replace_file(path('src'), path('dst'))
    assert errinfo.value.errno == errno.ENOENT

    with pytest.raises(OSError) as errinfo2:
        library.python.fs.replace(path('src'), path('dst2'))
    assert errinfo2.value.errno == errno.ENOENT


@in_env
def test_replace_file_exists(path):
    mkfile(path('src'), 'SRC')
    mkfile(path('dst'), 'DST')
    library.python.fs.replace_file(path('src'), path('dst'))
    assert not os.path.isfile(path('src'))
    assert os.path.isfile(path('dst'))
    assert file_data(path('dst')) == 'SRC'

    mkfile(path('src'), 'SRC')
    mkfile(path('dst2'), 'DST')
    library.python.fs.replace(path('src'), path('dst2'))
    assert not os.path.isfile(path('src'))
    assert os.path.isfile(path('dst2'))
    assert file_data(path('dst2')) == 'SRC'


@in_env
def test_replace_file_exists_dir_empty(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.replace_file(path('src'), path('dst'))
    assert errinfo.value.errno in (errno.EISDIR, errno.EACCES)
    assert os.path.isfile(path('src'))
    assert os.path.isdir(path('dst'))
    assert not os.path.isfile(path('dst/src'))


@in_env
def test_replace_file_exists_dir_empty_overwrite(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst'))
    library.python.fs.replace(path('src'), path('dst'))
    assert not os.path.isfile(path('src'))
    assert os.path.isfile(path('dst'))
    assert file_data(path('dst')) == 'SRC'


@in_env
def test_replace_file_exists_dir_nonempty(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst'))
    mkfile(path('dst/dst_file'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.replace_file(path('src'), path('dst'))
    assert errinfo.value.errno in (errno.EISDIR, errno.EACCES)
    assert os.path.isfile(path('src'))
    assert os.path.isdir(path('dst'))
    assert os.path.isfile(path('dst/dst_file'))
    assert not os.path.isfile(path('dst/src'))


@in_env
def test_replace_file_exists_dir_nonempty_overwrite(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst'))
    mkfile(path('dst/dst_file'))
    library.python.fs.replace(path('src'), path('dst'))
    assert not os.path.isfile(path('src'))
    assert os.path.isfile(path('dst'))
    assert file_data(path('dst')) == 'SRC'


@in_env
def test_replace_dir(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    library.python.fs.replace(path('src'), path('dst'))
    assert not os.path.isdir(path('src'))
    assert os.path.isdir(path('dst'))
    assert os.path.isfile(path('dst/src_file'))


@in_env
def test_replace_dir_exists_empty(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    os.mkdir(path('dst'))
    library.python.fs.replace(path('src'), path('dst'))
    assert not os.path.isdir(path('src'))
    assert os.path.isdir(path('dst'))
    assert os.path.isfile(path('dst/src_file'))


@in_env
def test_replace_dir_exists_nonempty(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    os.mkdir(path('dst'))
    mkfile(path('dst/dst_file'))
    library.python.fs.replace(path('src'), path('dst'))
    assert not os.path.isdir(path('src'))
    assert os.path.isdir(path('dst'))
    assert os.path.isfile(path('dst/src_file'))
    assert not os.path.isfile(path('dst/dst_file'))


@in_env
def test_replace_dir_exists_file(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    mkfile(path('dst'), 'DST')
    library.python.fs.replace(path('src'), path('dst'))
    assert not os.path.isdir(path('src'))
    assert os.path.isdir(path('dst'))
    assert os.path.isfile(path('dst/src_file'))


@in_env
def test_remove_file(path):
    mkfile(path('path'))
    library.python.fs.remove_file(path('path'))
    assert not os.path.exists(path('path'))


@in_env
def test_remove_file_no(path):
    with pytest.raises(OSError) as errinfo:
        library.python.fs.remove_file(path('path'))
    assert errinfo.value.errno == errno.ENOENT


@in_env
def test_remove_file_exists_dir(path):
    os.mkdir(path('path'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.remove_file(path('path'))
    assert errinfo.value.errno in (errno.EISDIR, errno.EACCES)
    assert os.path.isdir(path('path'))


@in_env
def test_remove_dir(path):
    os.mkdir(path('path'))
    library.python.fs.remove_dir(path('path'))
    assert not os.path.exists(path('path'))


@in_env
def test_remove_dir_no(path):
    with pytest.raises(OSError) as errinfo:
        library.python.fs.remove_dir(path('path'))
    assert errinfo.value.errno == errno.ENOENT


@in_env
def test_remove_dir_exists_file(path):
    mkfile(path('path'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.remove_dir(path('path'))
    assert errinfo.value.errno in (errno.ENOTDIR, errno.EINVAL)
    assert os.path.isfile(path('path'))


@in_env
def test_remove_tree(path):
    mktree_example(path, 'path')
    library.python.fs.remove_tree(path('path'))
    assert not os.path.exists(path('path'))


@in_env
def test_remove_tree_empty(path):
    os.mkdir(path('path'))
    library.python.fs.remove_tree(path('path'))
    assert not os.path.exists(path('path'))


@in_env
def test_remove_tree_file(path):
    mkfile(path('path'))
    library.python.fs.remove_tree(path('path'))
    assert not os.path.exists(path('path'))


@in_env
def test_remove_tree_no(path):
    with pytest.raises(OSError) as errinfo:
        library.python.fs.remove_tree(path('path'))
    assert errinfo.value.errno == errno.ENOENT


@in_env
def test_remove_tree_safe(path):
    library.python.fs.remove_tree_safe(path('path'))


@in_env
def test_ensure_removed(path):
    library.python.fs.ensure_removed(path('path'))


@in_env
def test_ensure_removed_exists(path):
    os.makedirs(path('dir/subdir'))
    library.python.fs.ensure_removed(path('dir'))
    assert not os.path.exists(path('dir'))


@in_env
def test_ensure_removed_exists_precise(path):
    os.makedirs(path('dir/subdir'))
    library.python.fs.ensure_removed(path('dir/subdir'))
    assert os.path.exists(path('dir'))
    assert not os.path.exists(path('dir/subdir'))


@in_env
def test_hardlink_file(path):
    mkfile(path('src'), 'SRC')
    library.python.fs.hardlink(path('src'), path('dst'))
    assert os.path.isfile(path('src'))
    assert os.path.isfile(path('dst'))
    assert file_data(path('dst')) == 'SRC'
    assert inodes_unsupported() or inodes_equal(path('src'), path('dst'))


@in_env
def test_hardlink_file_no_src(path):
    with pytest.raises(OSError) as errinfo:
        library.python.fs.hardlink(path('src'), path('dst'))
    assert errinfo.value.errno == errno.ENOENT


@in_env
def test_hardlink_file_exists(path):
    mkfile(path('src'), 'SRC')
    mkfile(path('dst'), 'DST')
    with pytest.raises(OSError) as errinfo:
        library.python.fs.hardlink(path('src'), path('dst'))
    assert errinfo.value.errno == errno.EEXIST
    assert os.path.isfile(path('src'))
    assert os.path.isfile(path('dst'))
    assert file_data(path('dst')) == 'DST'
    assert inodes_unsupported() or not inodes_equal(path('src'), path('dst'))


@in_env
def test_hardlink_file_exists_dir(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.hardlink(path('src'), path('dst'))
    assert errinfo.value.errno == errno.EEXIST
    assert os.path.isfile(path('src'))
    assert os.path.isdir(path('dst'))
    assert not os.path.isfile(path('dst/src'))


@in_env
def test_hardlink_dir(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.hardlink(path('src'), path('dst'))
    assert errinfo.value.errno in (errno.EPERM, errno.EACCES)
    assert os.path.isdir(path('src'))
    assert not os.path.isdir(path('dst'))


@pytest.mark.skipif(library.python.windows.on_win(), reason='Symlinks disabled on Windows')
@in_env
def test_symlink_file(path):
    mkfile(path('src'), 'SRC')
    library.python.fs.symlink(path('src'), path('dst'))
    assert os.path.isfile(path('src'))
    assert os.path.isfile(path('dst'))
    assert os.path.islink(path('dst'))
    assert file_data(path('dst')) == 'SRC'


@pytest.mark.skipif(library.python.windows.on_win(), reason='Symlinks disabled on Windows')
@in_env
def test_symlink_file_no_src(path):
    library.python.fs.symlink(path('src'), path('dst'))
    assert not os.path.isfile(path('src'))
    assert not os.path.isfile(path('dst'))
    assert os.path.islink(path('dst'))


@pytest.mark.skipif(library.python.windows.on_win(), reason='Symlinks disabled on Windows')
@in_env
def test_symlink_file_exists(path):
    mkfile(path('src'), 'SRC')
    mkfile(path('dst'), 'DST')
    with pytest.raises(OSError) as errinfo:
        library.python.fs.symlink(path('src'), path('dst'))
    assert errinfo.value.errno == errno.EEXIST
    assert os.path.isfile(path('src'))
    assert os.path.isfile(path('dst'))
    assert not os.path.islink(path('dst'))
    assert file_data(path('dst')) == 'DST'


@pytest.mark.skipif(library.python.windows.on_win(), reason='Symlinks disabled on Windows')
@in_env
def test_symlink_file_exists_dir(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.symlink(path('src'), path('dst'))
    assert errinfo.value.errno == errno.EEXIST
    assert os.path.isfile(path('src'))
    assert os.path.isdir(path('dst'))
    assert not os.path.islink(path('dst'))
    assert not os.path.isfile(path('dst/src'))


@pytest.mark.skipif(library.python.windows.on_win(), reason='Symlinks disabled on Windows')
@in_env
def test_symlink_dir(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    library.python.fs.symlink(path('src'), path('dst'))
    assert os.path.isdir(path('src'))
    assert os.path.isdir(path('dst'))
    assert os.path.islink(path('dst'))
    assert os.path.isfile(path('dst/src_file'))


@pytest.mark.skipif(library.python.windows.on_win(), reason='Symlinks disabled on Windows')
@in_env
def test_symlink_dir_no_src(path):
    library.python.fs.symlink(path('src'), path('dst'))
    assert not os.path.isdir(path('src'))
    assert not os.path.isdir(path('dst'))
    assert os.path.islink(path('dst'))


@pytest.mark.skipif(library.python.windows.on_win(), reason='Symlinks disabled on Windows')
@in_env
def test_symlink_dir_exists(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    os.mkdir(path('dst'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.symlink(path('src'), path('dst'))
    assert errinfo.value.errno == errno.EEXIST
    assert os.path.isdir(path('src'))
    assert os.path.isdir(path('dst'))
    assert not os.path.islink(path('dst'))
    assert not os.path.isfile(path('dst/src_file'))


@pytest.mark.skipif(library.python.windows.on_win(), reason='Symlinks disabled on Windows')
@in_env
def test_symlink_dir_exists_file(path):
    os.mkdir(path('src'))
    mkfile(path('src/src_file'))
    mkfile(path('dst'), 'DST')
    with pytest.raises(OSError) as errinfo:
        library.python.fs.symlink(path('src'), path('dst'))
    assert errinfo.value.errno == errno.EEXIST
    assert os.path.isdir(path('src'))
    assert os.path.isfile(path('dst'))
    assert not os.path.islink(path('dst'))


@in_env
def test_hardlink_tree(path):
    mktree_example(path, 'src')
    library.python.fs.hardlink_tree(path('src'), path('dst'))
    assert trees_equal(path('src'), path('dst'))


@in_env
def test_hardlink_tree_empty(path):
    os.mkdir(path('src'))
    library.python.fs.hardlink_tree(path('src'), path('dst'))
    assert trees_equal(path('src'), path('dst'))


@in_env
def test_hardlink_tree_file(path):
    mkfile(path('src'), 'SRC')
    library.python.fs.hardlink_tree(path('src'), path('dst'))
    assert trees_equal(path('src'), path('dst'))


@in_env
def test_hardlink_tree_no_src(path):
    with pytest.raises(OSError) as errinfo:
        library.python.fs.hardlink_tree(path('src'), path('dst'))
    assert errinfo.value.errno == errno.ENOENT


@in_env
def test_hardlink_tree_exists(path):
    mktree_example(path, 'src')
    os.mkdir(path('dst_dir'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.hardlink_tree(path('src'), path('dst_dir'))
    assert errinfo.value.errno == errno.EEXIST
    mkfile(path('dst_file'), 'DST')
    with pytest.raises(OSError) as errinfo:
        library.python.fs.hardlink_tree(path('src'), path('dst_file'))
    assert errinfo.value.errno == errno.EEXIST


@in_env
def test_hardlink_tree_file_exists(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst_dir'))
    with pytest.raises(OSError) as errinfo:
        library.python.fs.hardlink_tree(path('src'), path('dst_dir'))
    assert errinfo.value.errno == errno.EEXIST
    mkfile(path('dst_file'), 'DST')
    with pytest.raises(OSError) as errinfo:
        library.python.fs.hardlink_tree(path('src'), path('dst_file'))
    assert errinfo.value.errno == errno.EEXIST


@in_env
def test_copy_file(path):
    mkfile(path('src'), 'SRC')
    library.python.fs.copy_file(path('src'), path('dst'))
    assert os.path.isfile(path('src'))
    assert os.path.isfile(path('dst'))
    assert file_data(path('dst')) == 'SRC'


@in_env
def test_copy_file_no_src(path):
    with pytest.raises(EnvironmentError):
        library.python.fs.copy_file(path('src'), path('dst'))


@in_env
def test_copy_file_exists(path):
    mkfile(path('src'), 'SRC')
    mkfile(path('dst'), 'DST')
    library.python.fs.copy_file(path('src'), path('dst'))
    assert os.path.isfile(path('src'))
    assert os.path.isfile(path('dst'))
    assert file_data(path('dst')) == 'SRC'


@in_env
def test_copy_file_exists_dir_empty(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst'))
    with pytest.raises(EnvironmentError):
        library.python.fs.copy_file(path('src'), path('dst'))
    assert os.path.isfile(path('src'))
    assert os.path.isdir(path('dst'))
    assert not os.path.isfile(path('dst/src'))


@in_env
def test_copy_file_exists_dir_nonempty(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst'))
    mkfile(path('dst/dst_file'))
    with pytest.raises(EnvironmentError):
        library.python.fs.copy_file(path('src'), path('dst'))
    assert os.path.isfile(path('src'))
    assert os.path.isdir(path('dst'))
    assert os.path.isfile(path('dst/dst_file'))
    assert not os.path.isfile(path('dst/src'))


@in_env
def test_copy_tree(path):
    mktree_example(path, 'src')
    library.python.fs.copy_tree(path('src'), path('dst'))
    assert trees_equal(path('src'), path('dst'))


@in_env
def test_copy_tree_empty(path):
    os.mkdir(path('src'))
    library.python.fs.copy_tree(path('src'), path('dst'))
    assert trees_equal(path('src'), path('dst'))


@in_env
def test_copy_tree_file(path):
    mkfile(path('src'), 'SRC')
    library.python.fs.copy_tree(path('src'), path('dst'))
    assert trees_equal(path('src'), path('dst'))


@in_env
def test_copy_tree_no_src(path):
    with pytest.raises(EnvironmentError):
        library.python.fs.copy_tree(path('src'), path('dst'))


@in_env
def test_copy_tree_exists(path):
    mktree_example(path, 'src')
    os.mkdir(path('dst_dir'))
    with pytest.raises(EnvironmentError):
        library.python.fs.copy_tree(path('src'), path('dst_dir'))
    mkfile(path('dst_file'), 'DST')
    with pytest.raises(EnvironmentError):
        library.python.fs.copy_tree(path('src'), path('dst_file'))


@in_env
def test_copy_tree_file_exists(path):
    mkfile(path('src'), 'SRC')
    os.mkdir(path('dst_dir'))
    with pytest.raises(EnvironmentError):
        library.python.fs.copy_tree(path('src'), path('dst_dir'))
    mkfile(path('dst_file'), 'DST')
    library.python.fs.copy_tree(path('src'), path('dst_file'))
    assert trees_equal(path('src'), path('dst_file'))


@in_env
def test_read_file(path):
    mkfile(path('src'), 'SRC')
    assert library.python.fs.read_file(path('src')).decode(library.python.strings.fs_encoding()) == 'SRC'
    assert library.python.fs.read_file(path('src'), binary=False) == 'SRC'


@in_env
def test_read_file_empty(path):
    mkfile(path('src'))
    assert library.python.fs.read_file(path('src')).decode(library.python.strings.fs_encoding()) == ''
    assert library.python.fs.read_file(path('src'), binary=False) == ''


@in_env
def test_read_file_multiline(path):
    mkfile(path('src'), 'SRC line 1\nSRC line 2\n')
    assert (
        library.python.fs.read_file(path('src')).decode(library.python.strings.fs_encoding())
        == 'SRC line 1\nSRC line 2\n'
    )
    assert library.python.fs.read_file(path('src'), binary=False) == 'SRC line 1\nSRC line 2\n'


@in_env
def test_read_file_multiline_crlf(path):
    mkfile(path('src'), 'SRC line 1\r\nSRC line 2\r\n')
    assert (
        library.python.fs.read_file(path('src')).decode(library.python.strings.fs_encoding())
        == 'SRC line 1\r\nSRC line 2\r\n'
    )
    if library.python.windows.on_win() or six.PY3:  # universal newlines are by default in text mode in python3
        assert library.python.fs.read_file(path('src'), binary=False) == 'SRC line 1\nSRC line 2\n'
    else:
        assert library.python.fs.read_file(path('src'), binary=False) == 'SRC line 1\r\nSRC line 2\r\n'


@in_env
def test_read_file_unicode(path):
    s = u'АБВ'
    mkfile(path('src'), s.encode('utf-8'))
    mkfile(path('src_cp1251'), s.encode('cp1251'))
    assert library.python.fs.read_file_unicode(path('src')) == s
    assert library.python.fs.read_file_unicode(path('src_cp1251'), enc='cp1251') == s
    assert library.python.fs.read_file_unicode(path('src'), binary=False) == s
    assert library.python.fs.read_file_unicode(path('src_cp1251'), binary=False, enc='cp1251') == s


@in_env
def test_read_file_unicode_empty(path):
    mkfile(path('src'))
    mkfile(path('src_cp1251'))
    assert library.python.fs.read_file_unicode(path('src')) == ''
    assert library.python.fs.read_file_unicode(path('src_cp1251'), enc='cp1251') == ''
    assert library.python.fs.read_file_unicode(path('src'), binary=False) == ''
    assert library.python.fs.read_file_unicode(path('src_cp1251'), binary=False, enc='cp1251') == ''


@in_env
def test_read_file_unicode_multiline(path):
    s = u'АБВ\nИ еще\n'
    mkfile(path('src'), s.encode('utf-8'))
    mkfile(path('src_cp1251'), s.encode('cp1251'))
    assert library.python.fs.read_file_unicode(path('src')) == s
    assert library.python.fs.read_file_unicode(path('src_cp1251'), enc='cp1251') == s
    assert library.python.fs.read_file_unicode(path('src'), binary=False) == s
    assert library.python.fs.read_file_unicode(path('src_cp1251'), binary=False, enc='cp1251') == s


@in_env
def test_read_file_unicode_multiline_crlf(path):
    s = u'АБВ\r\nИ еще\r\n'
    mkfile(path('src'), s.encode('utf-8'))
    mkfile(path('src_cp1251'), s.encode('cp1251'))
    assert library.python.fs.read_file_unicode(path('src')) == s
    assert library.python.fs.read_file_unicode(path('src_cp1251'), enc='cp1251') == s
    if library.python.windows.on_win() or six.PY3:  # universal newlines are by default in text mode in python3
        assert library.python.fs.read_file_unicode(path('src'), binary=False) == u'АБВ\nИ еще\n'
        assert library.python.fs.read_file_unicode(path('src_cp1251'), binary=False, enc='cp1251') == u'АБВ\nИ еще\n'
    else:
        assert library.python.fs.read_file_unicode(path('src'), binary=False) == s
        assert library.python.fs.read_file_unicode(path('src_cp1251'), binary=False, enc='cp1251') == s


@in_env
def test_write_file(path):
    library.python.fs.write_file(path('src'), 'SRC')
    assert file_data(path('src')) == 'SRC'
    library.python.fs.write_file(path('src2'), 'SRC', binary=False)
    assert file_data(path('src2')) == 'SRC'


@in_env
def test_write_file_empty(path):
    library.python.fs.write_file(path('src'), '')
    assert file_data(path('src')) == ''
    library.python.fs.write_file(path('src2'), '', binary=False)
    assert file_data(path('src2')) == ''


@in_env
def test_write_file_multiline(path):
    library.python.fs.write_file(path('src'), 'SRC line 1\nSRC line 2\n')
    assert file_data(path('src')) == 'SRC line 1\nSRC line 2\n'
    library.python.fs.write_file(path('src2'), 'SRC line 1\nSRC line 2\n', binary=False)
    if library.python.windows.on_win():
        assert file_data(path('src2')) == 'SRC line 1\r\nSRC line 2\r\n'
    else:
        assert file_data(path('src2')) == 'SRC line 1\nSRC line 2\n'


@in_env
def test_write_file_multiline_crlf(path):
    library.python.fs.write_file(path('src'), 'SRC line 1\r\nSRC line 2\r\n')
    assert file_data(path('src')) == 'SRC line 1\r\nSRC line 2\r\n'
    library.python.fs.write_file(path('src2'), 'SRC line 1\r\nSRC line 2\r\n', binary=False)
    if library.python.windows.on_win():
        assert file_data(path('src2')) == 'SRC line 1\r\r\nSRC line 2\r\r\n'
    else:
        assert file_data(path('src2')) == 'SRC line 1\r\nSRC line 2\r\n'


@in_env
def test_get_file_size(path):
    mkfile(path('one.txt'), '22')
    assert library.python.fs.get_file_size(path('one.txt')) == 2


@in_env
def test_get_file_size_empty(path):
    mkfile(path('one.txt'))
    assert library.python.fs.get_file_size(path('one.txt')) == 0


@in_env
def test_get_tree_size(path):
    os.makedirs(path('deeper'))
    mkfile(path('one.txt'), '1')
    mkfile(path('deeper/two.txt'), '22')
    assert library.python.fs.get_tree_size(path('one.txt')) == 1
    assert library.python.fs.get_tree_size(path('')) == 1
    assert library.python.fs.get_tree_size(path(''), recursive=True) == 3


@pytest.mark.skipif(library.python.windows.on_win(), reason='Symlinks disabled on Windows')
@in_env
def test_get_tree_size_dangling_symlink(path):
    os.makedirs(path('deeper'))
    mkfile(path('one.txt'), '1')
    mkfile(path('deeper/two.txt'), '22')
    os.symlink(path('deeper/two.txt'), path("deeper/link.txt"))
    os.remove(path('deeper/two.txt'))
    # does not fail
    assert library.python.fs.get_tree_size(path(''), recursive=True) == 1


@pytest.mark.skipif(not library.python.windows.on_win(), reason='Test hardlinks on windows')
def test_hardlink_or_copy():
    max_allowed_hard_links = 1023

    def run(hardlink_function, dir):
        src = r"test.txt"
        with open(src, "w") as f:
            f.write("test")
        for i in range(max_allowed_hard_links + 1):
            hardlink_function(src, os.path.join(dir, "{}.txt".format(i)))

    dir1 = library.python.fs.create_dirs("one")
    with pytest.raises(WindowsError) as e:
        run(library.python.fs.hardlink, dir1)
    assert e.value.winerror == 1142
    assert len(os.listdir(dir1)) == max_allowed_hard_links

    dir2 = library.python.fs.create_dirs("two")
    run(library.python.fs.hardlink_or_copy, dir2)
    assert len(os.listdir(dir2)) == max_allowed_hard_links + 1


def test_remove_tree_unicode():
    path = u"test_remove_tree_unicode/русский".encode("utf-8")
    os.makedirs(path)
    library.python.fs.remove_tree(six.text_type("test_remove_tree_unicode"))
    assert not os.path.exists("test_remove_tree_unicode")


def test_remove_tree_safe_unicode():
    path = u"test_remove_tree_safe_unicode/русский".encode("utf-8")
    os.makedirs(path)
    library.python.fs.remove_tree_safe(six.text_type("test_remove_tree_safe_unicode"))
    assert not os.path.exists("test_remove_tree_safe_unicode")


def test_copy_tree_custom_copy_function():
    library.python.fs.create_dirs("test_copy_tree_src/deepper/inner")
    library.python.fs.write_file("test_copy_tree_src/deepper/deepper.txt", "deepper.txt")
    library.python.fs.write_file("test_copy_tree_src/deepper/inner/inner.txt", "inner.txt")
    copied = []

    def copy_function(src, dst):
        shutil.copy2(src, dst)
        copied.append(dst)

    library.python.fs.copy_tree(
        "test_copy_tree_src", yatest.common.work_path("test_copy_tree_dst"), copy_function=copy_function
    )
    assert len(copied) == 2
    assert yatest.common.work_path("test_copy_tree_dst/deepper/deepper.txt") in copied
    assert yatest.common.work_path("test_copy_tree_dst/deepper/inner/inner.txt") in copied


def test_copy2():
    library.python.fs.symlink("non-existent", "link")
    library.python.fs.copy2("link", "link2", follow_symlinks=False)

    assert os.path.islink("link2")
    assert os.readlink("link2") == "non-existent"


def test_commonpath():
    pj = os.path.join
    pja = lambda *x: os.path.abspath(pj(*x))

    assert library.python.fs.commonpath(['a', 'b']) == ''
    assert library.python.fs.commonpath([pj('t', '1')]) == pj('t', '1')
    assert library.python.fs.commonpath([pj('t', '1'), pj('t', '2')]) == pj('t')
    assert library.python.fs.commonpath([pj('t', '1', '2'), pj('t', '1', '2')]) == pj('t', '1', '2')
    assert library.python.fs.commonpath([pj('t', '1', '1'), pj('t', '1', '2')]) == pj('t', '1')
    assert library.python.fs.commonpath([pj('t', '1', '1'), pj('t', '1', '2'), pj('t', '1', '3')]) == pj('t', '1')

    assert library.python.fs.commonpath([pja('t', '1', '1'), pja('t', '1', '2')]) == pja('t', '1')

    assert library.python.fs.commonpath({pj('t', '1'), pj('t', '2')}) == pj('t')
