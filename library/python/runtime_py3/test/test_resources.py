import importlib.resources as ir

from importlib.resources._common import get_resource_reader
from importlib.resources.abc import TraversalError

import pytest

from sitecustomize import ResfsTraversableResources


@pytest.mark.parametrize(
    "package, resource",
    (
        ("resources", "foo.txt"),
        ("resources.submodule", "bar.txt"),
    ),
)
def test_is_resource_good_path(package, resource):
    assert ir.is_resource(package, resource)


@pytest.mark.parametrize(
    "package, resource",
    (
        ("resources", "111.txt"),
        ("resources.submodule", "222.txt"),
    ),
)
def test_is_resource_missing(package, resource):
    assert not ir.is_resource(package, resource)


@pytest.mark.parametrize(
    "directory",
    (
        "data",
        "submodule",
    ),
)
def test_is_resource_subresource_directory(directory):
    # Directories are not resources.
    assert not ir.is_resource("resources", directory)


@pytest.mark.parametrize(
    "package, resource, expected",
    (
        ("resources", "foo.txt", b"bar"),
        ("resources.submodule", "bar.txt", b"foo"),
    ),
)
def test_read_binary_good_path(package, resource, expected):
    assert ir.read_binary(package, resource) == expected


def test_read_binary_missing():
    with pytest.raises(TraversalError):
        ir.read_binary("resources", "111.txt")


@pytest.mark.parametrize(
    "package, resource, expected",
    (
        ("resources", "foo.txt", "bar"),
        ("resources.submodule", "bar.txt", "foo"),
    ),
)
def test_read_text_good_path(package, resource, expected):
    assert ir.read_text(package, resource) == expected


def test_read_text_missing():
    with pytest.raises(TraversalError):
        ir.read_text("resources", "111.txt")


@pytest.mark.parametrize(
    "package, expected",
    (
        ("resources", ["data", "submodule", "foo.txt"]),
        ("resources.submodule", ["bar.txt"]),
    ),
)
def test_contents_good_path(package, expected):
    assert sorted(ir.contents(package)) == sorted(expected)


def test_files_joinpath():
    assert ir.files("resources") / "submodule"
    assert ir.files("resources") / "foo.txt"
    assert ir.files("resources") / "data" / "my_data"
    assert ir.files("resources") / "submodule" / "bar.txt"
    assert ir.files("resources.submodule") / "bar.txt"


@pytest.mark.parametrize(
    "package, resource, expected",
    (
        ("resources", "foo.txt", b"bar"),
        ("resources", "data/my_data", b"data"),
        ("resources", "submodule/bar.txt", b"foo"),
        ("resources.submodule", "bar.txt", b"foo"),
    ),
)
def test_files_read_bytes(package, resource, expected):
    assert (ir.files(package) / resource).read_bytes() == expected


@pytest.mark.parametrize(
    "package, resource, expected",
    (
        ("resources", "foo.txt", "bar"),
        ("resources", "data/my_data", "data"),
        ("resources", "submodule/bar.txt", "foo"),
        ("resources.submodule", "bar.txt", "foo"),
    ),
)
def test_files_read_text(package, resource, expected):
    assert (ir.files(package) / resource).read_text() == expected


@pytest.mark.parametrize(
    "package, expected",
    (
        ("resources", ("foo.txt", "data", "submodule")),
        ("resources.submodule", ("bar.txt",)),
    ),
)
def test_files_iterdir(package, expected):
    assert tuple(resource.name for resource in ir.files(package).iterdir()) == expected


@pytest.mark.parametrize(
    "package, expected",
    (
        ("resources", ("data", "foo.txt", "submodule")),
        ("resources.submodule", ("bar.txt",)),
    ),
)
def test_files_iterdir_with_sort(package, expected):
    assert tuple(resource.name for resource in sorted(ir.files(package).iterdir())) == expected


def test_get_resource_reader():
    import resources

    reader = get_resource_reader(resources)
    assert isinstance(reader, ResfsTraversableResources)

    assert reader.is_resource("foo.txt") is True
    assert reader.is_resource("submodule/bar.txt") is True
    assert reader.is_resource("notfound.txt") is False
    assert reader.is_resource("submodule") is False

    with pytest.raises(FileNotFoundError) as ex:
        reader.resource_path("foo.txt")
    assert str(ex.value) == "foo.txt"

    with reader.open_resource("foo.txt") as f:
        assert f.read() == b"bar"
    with pytest.raises(FileNotFoundError) as ex:
        reader.open_resource("notfound.txt")
    assert str(ex.value) == "resfs/file/library/python/runtime_py3/test/resources/notfound.txt"

    assert tuple(reader.contents()) == ("foo.txt", "data", "submodule")
