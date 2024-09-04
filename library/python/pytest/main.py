import os
import sys
import time

import __res

FORCE_EXIT_TESTSFAILED_ENV = 'FORCE_EXIT_TESTSFAILED'


def main():
    import library.python.pytest.context as context

    context.Ctx["YA_PYTEST_START_TIMESTAMP"] = time.time()

    profile = None
    if '--profile-pytest' in sys.argv:
        sys.argv.remove('--profile-pytest')

        import pstats
        import cProfile

        profile = cProfile.Profile()
        profile.enable()

    # Reset influencing env. vars
    # For more info see library/python/testing/yatest_common/yatest/common/errors.py
    if FORCE_EXIT_TESTSFAILED_ENV in os.environ:
        del os.environ[FORCE_EXIT_TESTSFAILED_ENV]

    if "Y_PYTHON_CLEAR_ENTRY_POINT" in os.environ:
        if "Y_PYTHON_ENTRY_POINT" in os.environ:
            del os.environ["Y_PYTHON_ENTRY_POINT"]
        del os.environ["Y_PYTHON_CLEAR_ENTRY_POINT"]

    listing_mode = '--collect-only' in sys.argv
    yatest_runner = os.environ.get('YA_TEST_RUNNER') == '1'

    import pytest

    import library.python.pytest.plugins.collection as collection
    import library.python.pytest.plugins.ya as ya
    import library.python.pytest.plugins.conftests as conftests

    import _pytest.assertion
    from _pytest.monkeypatch import MonkeyPatch
    from . import rewrite

    m = MonkeyPatch()
    m.setattr(_pytest.assertion.rewrite, "AssertionRewritingHook", rewrite.AssertionRewritingHook)

    # see https://st.yandex-team.ru/DEVTOOLSSUPPORT-50337
    m.setattr(_pytest.assertion.truncate, "DEFAULT_MAX_LINES", 16)
    m.setattr(_pytest.assertion.truncate, "DEFAULT_MAX_CHARS", 32 * 80)

    prefix = '__tests__.'

    test_modules = [
        # fmt: off
        name[len(prefix) :]
        for name in sys.extra_modules
        if name.startswith(prefix) and not name.endswith('.conftest')
        # fmt: on
    ]

    doctest_packages = __res.find("PY_DOCTEST_PACKAGES") or ""
    if isinstance(doctest_packages, bytes):
        doctest_packages = doctest_packages.decode('utf-8')
    doctest_packages = doctest_packages.split()

    def is_doctest_module(name):
        for package in doctest_packages:
            if name == package or name.startswith(str(package) + "."):
                return True
        return False

    doctest_modules = [
        # fmt: off
        name
        for name in sys.extra_modules
        if is_doctest_module(name)
        # fmt: on
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
    rc = pytest.main(
        plugins=[
            collection.CollectionPlugin(test_modules, doctest_modules),
            ya,
            conftests,
        ]
    )

    if rc == 5:
        # don't care about EXIT_NOTESTSCOLLECTED
        rc = 0

    if rc == 1 and yatest_runner and not listing_mode and not os.environ.get(FORCE_EXIT_TESTSFAILED_ENV) == '1':
        # XXX it's place for future improvements
        # Test wrapper should terminate with 0 exit code if there are common test failures
        # and report it with trace-file machinery.
        # However, there are several case when we don't want to suppress exit_code:
        #  - listing machinery doesn't use trace-file currently and rely on stdout and exit_code
        #  - RestartTestException and InfrastructureException required non-zero exit_code to be processes correctly
        rc = 0

    if profile:
        profile.disable()
        ps = pstats.Stats(profile, stream=sys.stderr).sort_stats('cumulative')
        ps.print_stats()
        if '--output-dir' in sys.argv:
            output_dir = sys.argv[sys.argv.index('--output-dir') + 1]
            prof_filename = os.path.join(output_dir, 'pytest.profile')
            ps.dump_stats(prof_filename)

            try:
                import gprof2dot
            except ImportError as e:
                sys.stderr.write("Failed to generate call graph: {}\n".format(e))
                gprof2dot = None

            if gprof2dot:
                import shlex

                dot_filename = os.path.join(output_dir, 'pytest.profile.dot')
                args = [
                    prof_filename,
                    '--format=pstats',
                    '--output={}'.format(dot_filename),
                ]
                if 'PYTEST_GPROF2DOT_ARGS' in os.environ:
                    x = os.environ['PYTEST_GPROF2DOT_ARGS']
                    args.extend(shlex.split(x))

                gprof2dot.main(argv=args)

    sys.exit(rc)


if __name__ == '__main__':
    main()
