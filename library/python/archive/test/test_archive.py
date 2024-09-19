# coding=utf-8

import errno
import hashlib
import os
import shutil
import socket
import tarfile
import threading

import six

import pytest
import yatest.common

import library.python.archive as archive


if six.PY2:

    class PermissionError(Exception):
        pass


data_dir = yatest.common.build_path("library/python/archive/test/data")


def extract_tar(filename, dirname, strip_components=None, apply_mtime=False, entry_filter=None):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    return archive.extract_tar(filename, dirname, strip_components, apply_mtime=apply_mtime, entry_filter=entry_filter)


def test_get_filenames():
    filenames = archive.get_archive_filenames(os.path.join(data_dir, "sample.tar"))
    assert sorted(filenames) == sorted(["1.txt", "2/2.txt", "executable.sh"])


@pytest.mark.parametrize("apply_mtime", [True, False])
def test_extract_mtime(apply_mtime):
    out_tar = yatest.common.output_path("out_tar")
    extract_tar(os.path.join(data_dir, "sample.tar"), out_tar, apply_mtime=apply_mtime)
    with tarfile.open(os.path.join(data_dir, "sample.tar"), 'r') as tar:
        for fn in ["1.txt", "2/2.txt", "executable.sh"]:
            assert (os.path.getmtime(os.path.join(out_tar, fn)) == tar.getmember(fn).mtime) == apply_mtime


def test_extract_entry_filter():
    def _entry_filter(entry):
        return entry.pathname == "2/2.txt"

    out_tar = yatest.common.output_path("out_tar")
    extract_tar(os.path.join(data_dir, "sample.tar"), out_tar, entry_filter=_entry_filter)
    assert os.path.exists(os.path.join(out_tar, "2/2.txt"))
    for fn in ["1.txt", "executable.sh"]:
        assert not os.path.exists(os.path.join(out_tar, fn))


def test_is_empty():
    with_entries = os.path.join(data_dir, "sample.tar")
    wo_entries = yatest.common.output_path("empty.tar")
    archive.tar([], wo_entries)
    assert not archive.is_empty(with_entries)
    assert archive.is_empty(wo_entries)


def test_extract_tar():
    out_tar = yatest.common.output_path("out_tar")
    out_tar_gz = yatest.common.output_path("out_tar_gz")
    extract_tar(os.path.join(data_dir, "sample.tar"), out_tar)
    extract_tar(os.path.join(data_dir, "sample.tar.gz"), out_tar_gz)
    for d in [out_tar, out_tar_gz]:
        for f in ["1.txt", "2/2.txt", "executable.sh"]:
            assert os.path.exists(os.path.join(d, f))
        assert os.access(os.path.join(d, "executable.sh"), os.X_OK)


def test_extract_tar_no_slash_dir():
    # this tarball has entry `package` that is a directory
    # but its name does not have trailing slash
    out_color = yatest.common.output_path("out_color")
    extract_tar(os.path.join(data_dir, "color-0.1.2.tgz"), out_color)
    files = [
        "package",
        "package/LICENSE",
        "package/README.md",
        "package/lib",
        "package/package.json",
        "package/lib/index.d.ts",
        "package/lib/index.js",
    ]
    for f in files:
        assert os.path.exists(os.path.join(out_color, f))


@pytest.mark.parametrize(
    'components,expected',
    [
        (None, ["1.txt", "2", "executable.sh"]),
        (0, ["1.txt", "2", "executable.sh"]),
        (1, ["2.txt"]),
        (2, []),
    ],
)
def test_extract_tar_strip_components(components, expected):
    out_tar = yatest.common.output_path("out_tar")
    extract_tar(os.path.join(data_dir, "sample.tar"), out_tar, strip_components=components)
    files = set(os.listdir(out_tar))
    assert files == set(expected)


def test_extract_tar_fail_on_duplicates():
    out_tar = "out_tar"
    test_file_name = "test_file.txt"
    open(test_file_name, "w").close()
    tar_path = yatest.common.output_path("test_file_duplicates.tar")
    archive.tar(
        [
            (test_file_name, "dir1/file"),
            (test_file_name, "dir2/file"),
        ],
        tar_path,
    )

    # Extracting without strip_components doesn't raise an error
    extract_tar(tar_path, out_tar)

    with pytest.raises(Exception, match="duplicated"):
        extract_tar(tar_path, out_tar, strip_components=1)


def test_tar_add_files():
    archive._make_dirs("one")
    archive._make_dirs("two")

    with open("one/one.txt", "w") as f:
        f.write("one")

    with open("two/two.txt", "w") as f:
        f.write("two")
    tar_path = yatest.common.output_path("test_tar_multiple_paths.tar")
    archive.tar(
        [
            ("one/one.txt", "one.txt"),
            ("two/two.txt", "two.txt"),
            ("one/one.txt", "one/one.txt"),
            ("two/two.txt", "two/two.txt"),
        ],
        tar_path,
    )

    expected_tar_members = [
        "one.txt",
        "two.txt",
        "one/one.txt",
        "two/two.txt",
    ]

    with tarfile.open(tar_path) as tar:
        for m in expected_tar_members:
            assert tar.getmember(m)


def test_create_add_one_file():
    archive._make_dirs("test_create")

    with open("test_create/test.txt", "w") as f:
        f.write("test")

    tar_path = yatest.common.output_path("test_create.tar")
    archive.tar("test_create/test.txt", tar_path)

    expected_tar_members = ["test.txt"]

    with tarfile.open(tar_path) as tar:
        for m in expected_tar_members:
            assert tar.getmember(m)


def test_tar_add_dir():
    archive._make_dirs("test_add_dir/1/2/3")

    with open("test_add_dir/1/1.txt", "w") as f:
        f.write("1")
    with open("test_add_dir/1/2/2.txt", "w") as f:
        f.write("2")
    with open("test_add_dir/1/2/3/3.txt", "w") as f:
        f.write("3")

    tar_path = yatest.common.output_path("test_add_dir.tar")
    archive.tar("test_add_dir", tar_path)

    expected_tar_members = [
        "test_add_dir/1/1.txt",
        "test_add_dir/1/2/2.txt",
        "test_add_dir/1/2/3/3.txt",
    ]

    with tarfile.open(tar_path) as tar:
        for m in expected_tar_members:
            assert tar.getmember(m)

    tar_path = yatest.common.output_path("test_add_dir_2.tar")
    archive.tar([("test_add_dir", "root")], tar_path)

    expected_tar_members = ["root/1/1.txt", "root/1/2/2.txt", "root/1/2/3/3.txt"]

    with tarfile.open(tar_path) as tar:
        for m in expected_tar_members:
            assert tar.getmember(m)

    tar_path = yatest.common.output_path("test_add_dir_3.tar")
    archive.tar([("test_add_dir", ".")], tar_path)

    expected_tar_members = ["1/1.txt", "1/2/2.txt", "1/2/3/3.txt"]

    with tarfile.open(tar_path) as tar:
        for m in expected_tar_members:
            assert tar.getmember(m)


def test_check_archive():
    assert archive.check_tar(os.path.join(data_dir, "sample.tar"))
    assert archive.check_tar(os.path.join(data_dir, "sample.tar.gz"))


def test_unicode():
    path = "файл.txt"
    tar_path = yatest.common.output_path("{}.tar".format(path))
    with open(path, "w") as f:
        f.write("тест")
    archive.tar([path], tar_path)

    with tarfile.open(tar_path) as tar:
        assert tar.getmember(path)


def test_long_filename():
    # DEVTOOLS-3546
    root = yatest.common.output_path("archive")
    filename = os.path.join(root, os.sep.join([s * 100 for s in "01234567890"]), "file.txt")
    os.makedirs(os.path.dirname(filename))
    tar_path = yatest.common.output_path("long.tar")

    with tarfile.open(filename, "w:") as tar:
        tar.add(root)

    with open(filename, "w") as f:
        f.write("long")
    archive.tar([root], tar_path)

    with tarfile.open(tar_path) as tar:
        assert tar.getmember("archive/{}".format(os.path.relpath(filename, root)))


def test_non_regular_files():
    archive._make_dirs("non_regular_files")

    with open("non_regular_files/one.txt", "w") as f:
        f.write("one")

    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind("non_regular_files/socket")

    os.mkfifo("non_regular_files/fifo")

    tar_path = yatest.common.output_path("test_non_regular_files.tar")
    archive.tar("non_regular_files", tar_path)

    with tarfile.open(tar_path) as tar:
        tar.getmembers() == ["non_regular_files/one.txt"]


def list_files(path):
    return sorted(os.path.relpath(os.path.join(root, f), path) for root, _, files in os.walk(path) for f in files)


@pytest.mark.parametrize("filter_name", [archive.GZIP, archive.ZSTD])
@pytest.mark.parametrize("level", [archive.Compression.Fast, archive.Compression.Best, 3, None])
def test_compression(filter_name, level):
    archive._make_dirs("compression")

    with open("compression/one.txt", "w") as f:
        f.write("one")

    tar_path = yatest.common.output_path("compression.tar")
    archive.tar("compression", tar_path, compression_filter=filter_name, compression_level=level)

    extract_tar(tar_path, "out")
    assert list_files("out") == ["compression/one.txt"]


@pytest.mark.parametrize("compression_filter", [archive.ZSTD, archive.GZIP])
@pytest.mark.parametrize("compression_level", [None, 4, 8])
def test_stable_archive(compression_level, compression_filter):
    archive._make_dirs("stable")
    archive._make_dirs("stable/dir")
    with open("stable/stable.txt", "w") as f:
        f.write("pretty l" + ("o" * 1000) + "ng string")

    tar_path = yatest.common.output_path("test_stable.tar")

    def tar_and_get_hash(fixed_mtime):
        # change mtime
        os.utime("stable/stable.txt", (0, os.stat("stable/stable.txt").st_mtime + 1))

        archive.tar(
            "stable",
            tar_path,
            compression_filter=compression_filter,
            compression_level=compression_level,
            fixed_mtime=fixed_mtime,
        )
        with open(tar_path, "rb") as afile:
            return hashlib.md5(afile.read()).digest()

    hash1 = tar_and_get_hash(fixed_mtime=None)

    extract_tar(tar_path, "out")
    assert list_files("out") == ["stable/stable.txt"]

    # mtime changed
    assert hash1 != tar_and_get_hash(fixed_mtime=None)
    assert tar_and_get_hash(fixed_mtime=0) == tar_and_get_hash(fixed_mtime=0)


@pytest.mark.parametrize("comp_level", [None, 4])
@pytest.mark.parametrize("comp_filter", [archive.GZIP, archive.ZSTD])
def test_archive_to_file_descriptor(comp_filter, comp_level):
    archive._make_dirs("to_file_descriptor")

    with open("to_file_descriptor/one.txt", "w") as f:
        f.write("one")

    tar_path = yatest.common.output_path("to_file_descriptor.tar")
    rfd, wfd = os.pipe()

    def write_tar():
        archive.tar("to_file_descriptor", wfd, comp_filter, comp_level)
        os.close(wfd)

    th = threading.Thread(target=write_tar)
    th.daemon = True
    th.start()

    with open(tar_path, "wb") as afile:
        data = 1
        while data:
            data = os.read(rfd, 64 * 1024)
            afile.write(data)

    th.join()

    extract_tar(tar_path, "out")
    assert list_files("out") == ["to_file_descriptor/one.txt"]
    with open("out/to_file_descriptor/one.txt") as afile:
        assert afile.read() == "one"


@pytest.mark.parametrize("comp_level", [None, 4])
@pytest.mark.parametrize("comp_filter", [archive.GZIP, archive.ZSTD])
def test_archive_to_file_handle(comp_filter, comp_level):
    archive._make_dirs("to_file_handle")

    with open("to_file_handle/one.txt", "w") as f:
        f.write("one")

    tar_path = yatest.common.output_path("to_file_handle.tar")

    with open(tar_path, "w") as afile:
        archive.tar("to_file_handle", afile, comp_filter, comp_level)

    extract_tar(tar_path, "out")
    assert list_files("out") == ["to_file_handle/one.txt"]
    with open("out/to_file_handle/one.txt") as afile:
        assert afile.read() == "one"


def test_onerror_retry():
    archive._make_dirs("onerror_retry")

    with open("onerror_retry/one.txt", "w") as f:
        f.write("one")

    os.chmod("onerror_retry/one.txt", 0o0222)
    tar_path = yatest.common.output_path("onerror_retry.tar")

    def onerror(src, dst, exc_info):
        if six.PY2:
            assert exc_info[0] == IOError
        else:
            assert exc_info[0] == PermissionError
        assert exc_info[1].errno == errno.EACCES
        os.chmod(src, 0o0555)
        return True

    with open(tar_path, "w") as afile:
        archive.tar("onerror_retry", afile, onerror=onerror)

    with tarfile.open(tar_path) as tar:
        tar.getmembers() == ["onerror_retry/one.txt"]
        assert tar.extractfile("onerror_retry/one.txt").read() == b"one"


def test_onerror_noretry():
    archive._make_dirs("onerror_noretry")

    with open("onerror_noretry/one.txt", "w") as f:
        f.write("one")
    with open("onerror_noretry/two.txt", "w") as f:
        f.write("two")

    archive._make_dirs("onerror_noretry/dir")

    os.chmod("onerror_noretry/one.txt", 0o0222)
    os.chmod("onerror_noretry/dir", 0o0222)
    tar_path = yatest.common.output_path("onerror_noretry.tar")

    try:
        with open(tar_path, "w") as afile:
            archive.tar("onerror_noretry", afile, onerror=lambda *args: False)

        with tarfile.open(tar_path) as tar:
            tar.getmembers() == ["onerror_noretry/two.txt"]
    finally:
        os.chmod("onerror_noretry/one.txt", 0o0555)
        os.chmod("onerror_noretry/dir", 0o0555)


@pytest.mark.parametrize("comp_filter", [None, archive.GZIP, archive.ZSTD])
def test_get_filter_name(comp_filter):
    archive._make_dirs("get_filter_name")
    tar_path = yatest.common.output_path("onerror_noretry.tar")
    archive.tar("get_filter_name", tar_path, comp_filter)
    assert archive.get_archive_filter_name(tar_path) == comp_filter


def gen_dereference_dir(name):
    dirname = "dereference_{}".format(name)
    archive._make_dirs(dirname + "/inner")
    with open(dirname + "/outside.txt", "w") as f:
        f.write("one")

    os.symlink("../outside.txt", dirname + "/inner/link.txt")
    return dirname


def test_dereference_never():
    dirname = gen_dereference_dir("never")
    tar_path = yatest.common.output_path(dirname + ".tar")

    with open(tar_path, "w") as afile:
        archive.tar(dirname, afile, dereference=False)

    extract_tar(tar_path, "out")
    assert list_files("out") == [dirname + "/inner/link.txt", dirname + "/outside.txt"]
    assert os.path.islink("out/" + dirname + "/inner/link.txt")


def test_dereference_always():
    dirname = gen_dereference_dir("always")
    tar_path = yatest.common.output_path(dirname + ".tar")

    with open(tar_path, "w") as afile:
        archive.tar(dirname, afile, dereference=True)

    extract_tar(tar_path, "out")
    assert list_files("out") == [dirname + "/inner/link.txt", dirname + "/outside.txt"]
    with open("out/" + dirname + "/inner/link.txt") as afile:
        assert afile.read() == "one"
    assert not os.path.islink("out/" + dirname + "/inner/link.txt")


def test_dereference_always_dir():
    dirname = "dereference_dir"
    archive._make_dirs(dirname + "/inner")
    with open(dirname + "/inner/file.txt", "w") as f:
        f.write("one")

    os.symlink("inner", dirname + "/link")

    tar_path = yatest.common.output_path(dirname + ".tar")

    with open(tar_path, "w") as afile:
        archive.tar(dirname, afile, dereference=True)

    extract_tar(tar_path, "out")
    assert list_files("out") == [
        dirname + "/inner/file.txt",
        dirname + "/link/file.txt",
    ]
    with open("out/" + dirname + "/link/file.txt") as afile:
        assert afile.read() == "one"
    assert not os.path.islink("out/" + dirname + "/link")
