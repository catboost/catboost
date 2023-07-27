import importlib.resources as ir

import pytest


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


def test_is_resource_subresource_directory():
    # Directories are not resources.
    assert not ir.is_resource("resources", "submodule")


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
    with pytest.raises(FileNotFoundError):
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
    with pytest.raises(FileNotFoundError):
        ir.read_text("resources", "111.txt")


@pytest.mark.parametrize(
    "package, expected",
    (
        ("resources", ["submodule", "foo.txt"]),
        ("resources.submodule", ["bar.txt"]),
    ),
)
def test_contents_good_path(package, expected):
    assert sorted(ir.contents(package)) == sorted(expected)
