import os
import pytest

from build.plugins.lib.nots.package_manager.base.package_json import PackageJson, PackageJsonWorkspaceError


def test_get_workspace_dep_paths_ok():
    pj = PackageJson("/packages/foo/package.json")
    pj.data = {
        "dependencies": {
            "@yandex-int/bar": "workspace:../bar",
        },
        "devDependencies": {
            "@yandex-int/baz": "workspace:../baz",
        },
    }

    ws_dep_paths = pj.get_workspace_dep_paths()

    assert ws_dep_paths == [
        ("@yandex-int/bar", "../bar"),
        ("@yandex-int/baz", "../baz"),
    ]


def test_get_workspace_dep_paths_invalid_path():
    pj = PackageJson("/packages/foo/package.json")
    pj.data = {
        "dependencies": {
            "@yandex-int/bar": "workspace:*",
        },
    }

    with pytest.raises(PackageJsonWorkspaceError) as e:
        pj.get_workspace_dep_paths()

    assert str(e.value) == "Expected relative path specifier for workspace dependency, but got 'workspace:*' for @yandex-int/bar in /packages/foo/package.json"


def test_get_workspace_deps_ok():
    pj = PackageJson("/packages/foo/package.json")
    pj.data = {
        "dependencies": {
            "@yandex-int/bar": "workspace:../bar",
        },
        "devDependencies": {
            "@yandex-int/baz": "workspace:../baz",
        },
    }

    def load_mock(cls, path):
        p = PackageJson(path)
        p.data = {
            "name": "@yandex-int/{}".format(os.path.basename(os.path.dirname(path))),
        }
        return p
    PackageJson.load = classmethod(load_mock)

    ws_deps = pj.get_workspace_deps()

    assert len(ws_deps) == 2
    assert ws_deps[0].path == "/packages/bar/package.json"
    assert ws_deps[1].path == "/packages/baz/package.json"


def test_get_workspace_deps_with_wrong_name():
    pj = PackageJson("/packages/foo/package.json")
    pj.data = {
        "dependencies": {
            "@yandex-int/bar": "workspace:../bar",
        },
    }

    def load_mock(cls, path):
        p = PackageJson(path)
        p.data = {
            "name": "@shouldbe/{}".format(os.path.basename(os.path.dirname(path))),
        }
        return p
    PackageJson.load = classmethod(load_mock)

    with pytest.raises(PackageJsonWorkspaceError) as e:
        pj.get_workspace_deps()

    assert str(e.value) == "Workspace dependency name mismatch, found '@yandex-int/bar' instead of '@shouldbe/bar' in /packages/foo/package.json"


def test_get_workspace_map_ok():
    pj = PackageJson("/packages/foo/package.json")
    pj.data = {
        "dependencies": {
            "@yandex-int/bar": "workspace:../bar",
        },
    }

    def load_mock(cls, path):
        name = os.path.basename(os.path.dirname(path))
        p = PackageJson(path)
        p.data = {
            "name": "@yandex-int/{}".format(name),
            "dependencies": ({"@yandex-int/qux": "workspace:../qux"} if name == "bar" else {}),
        }
        return p
    PackageJson.load = classmethod(load_mock)

    ws_map = pj.get_workspace_map()

    assert len(ws_map) == 3
    assert ws_map["/packages/foo"][0].path == "/packages/foo/package.json"
    assert ws_map["/packages/foo"][1] == 0
    assert ws_map["/packages/bar"][0].path == "/packages/bar/package.json"
    assert ws_map["/packages/bar"][1] == 1
    assert ws_map["/packages/qux"][0].path == "/packages/qux/package.json"
    assert ws_map["/packages/qux"][1] == 2
