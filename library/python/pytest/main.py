import sys

import __res


def main():
    profile = None
    if '--profile-pytest' in sys.argv:
        sys.argv.remove('--profile-pytest')

        import pstats
        import cProfile
        profile = cProfile.Profile()
        profile.enable()

    import pytest

    import library.python.pytest.plugins.collection as collection
    import library.python.pytest.plugins.ya as ya
    import library.python.pytest.plugins.conftests as conftests

    import _pytest.assertion
    from _pytest.monkeypatch import MonkeyPatch
    from . import rewrite
    m = MonkeyPatch()
    m.setattr(_pytest.assertion.rewrite, "AssertionRewritingHook", rewrite.AssertionRewritingHook)

    prefix = '__tests__.'

    test_modules = [
        name[len(prefix):] for name in sys.extra_modules
        if name.startswith(prefix) and not name.endswith('.conftest')
    ]

    doctest_packages = (__res.find("PY_DOCTEST_PACKAGES") or "").split()

    def is_doctest_module(name):
        for package in doctest_packages:
            if name == package or name.startswith(package + "."):
                return True
        return False

    doctest_modules = [
        name for name in sys.extra_modules
        if is_doctest_module(name)
    ]

    def remove_user_site(paths):
        site_paths = ('site-packages', 'site-python')

        def is_site_path(path):
            for p in site_paths:
                if path.find(p) != -1:
                    return True
            return False

        new_paths = list(paths)
        for p in paths:
            if is_site_path(p):
                new_paths.remove(p)

        return new_paths

    sys.path = remove_user_site(sys.path)
    rc = pytest.main(plugins=[
        collection.CollectionPlugin(test_modules, doctest_modules),
        ya,
        conftests,
    ])

    if rc == 5:
        # don't care about EXIT_NOTESTSCOLLECTED
        rc = 0

    if profile:
        profile.disable()
        ps = pstats.Stats(profile, stream=sys.stderr).sort_stats('cumulative')
        ps.print_stats()

    sys.exit(rc)


if __name__ == '__main__':
    main()
