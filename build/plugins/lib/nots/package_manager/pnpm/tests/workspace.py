from build.plugins.lib.nots.package_manager.base import PackageJson
from build.plugins.lib.nots.package_manager.pnpm.workspace import PnpmWorkspace


def test_workspace_get_paths():
    ws = PnpmWorkspace(path="/packages/foo/pnpm-workspace.yaml")
    ws.packages = set([".", "../bar", "../../another/baz"])

    assert sorted(ws.get_paths()) == [
        "/another/baz",
        "/packages/bar",
        "/packages/foo",
    ]


def test_workspace_set_from_package_json():
    ws = PnpmWorkspace(path="/packages/foo/pnpm-workspace.yaml")
    pj = PackageJson(path="/packages/foo/package.json")
    pj.data = {
        "dependencies": {
            "@a/bar": "workspace:../bar",
        },
        "devDependencies": {
            "@a/baz": "workspace:../../another/baz",
        },
        "peerDependencies": {
            "@a/qux": "workspace:../../another/qux",
        },
        "optionalDependencies": {
            "@a/quux": "workspace:../../another/quux",
        }
    }

    ws.set_from_package_json(pj)

    assert sorted(ws.get_paths()) == [
        "/another/baz",
        "/another/quux",
        "/another/qux",
        "/packages/bar",
        "/packages/foo",
    ]


def test_workspace_merge():
    ws1 = PnpmWorkspace(path="/packages/foo/pnpm-workspace.yaml")
    ws1.packages = set([".", "../bar", "../../another/baz"])
    ws2 = PnpmWorkspace(path="/another/baz/pnpm-workspace.yaml")
    ws2.packages = set([".", "../qux"])

    ws1.merge(ws2)

    assert sorted(ws1.get_paths()) == [
        "/another/baz",
        "/another/qux",
        "/packages/bar",
        "/packages/foo",
    ]
