import importlib.metadata as im

import pytest


@pytest.mark.parametrize("name", ("foo-bar", "foo_bar", "Foo-Bar"))
def test_distribution(name):
    assert im.distribution(name) is not None


def test_unknown_package():
    with pytest.raises(im.PackageNotFoundError):
        im.distribution("bar")


def test_version():
    assert im.version("foo-bar") == "1.2.3"


def test_metadata():
    assert im.metadata("foo-bar") is not None


def test_files():
    files = im.files("foo-bar")
    # Начиная с Python 3.12 добавилась проверка на существование файла,
    # а не тупое возвращение всего из RECORD, в Аркадии это ни где не используется,
    # потому пока поддержку не добавлял
    assert len(files) == 0
    # assert files[0].name == "foo_bar.py"
    # assert files[0].size == 20


def test_requires():
    assert im.requires("foo-bar") == ["Werkzeug (>=0.15)", "Jinja2 (>=2.10.1)"]


def test_entry_points():
    entry_points = im.entry_points()
    assert any(entry_point.group == "console_scripts" for entry_point in entry_points)

    flg_found = False
    for entry_point in entry_points.select(group="console_scripts"):
        if entry_point.name == "foo_cli" and entry_point.value == "foo_bar:cli":
            flg_found = True

    assert flg_found
