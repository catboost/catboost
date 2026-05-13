"""
Bazel build rules for CatBoost.

Provides yunit_test() — a macro that creates a cc_test using CatBoost's
Y_UNIT_TEST_SUITE / Y_UNIT_TEST framework (library/cpp/testing/unittest).
"""

def yunit_test(name, srcs, deps = [], copts = [], **kwargs):
    """Creates a cc_test using CatBoost's Y_UNIT_TEST framework.

    The test binary uses the custom main() from library/cpp/testing/unittest
    which discovers and runs all Y_UNIT_TEST_SUITE / Y_UNIT_TEST tests.

    Args:
        name: Test target name.
        srcs: Test source files (typically *_ut.cpp).
        deps: Additional dependencies beyond yutil and the test framework.
        copts: Additional compiler options.
        **kwargs: Passed through to cc_test (e.g., size, timeout, tags).
    """
    native.cc_test(
        name = name,
        srcs = srcs,
        copts = ["-I."] + copts,
        deps = deps + [
            "//util:yutil",
            "//library/cpp/testing/unittest:unittest",
            "//library/cpp/testing/unittest_main:unittest_main",
        ],
        **kwargs
    )
