def test_conftest_plugin_order(pytestconfig):
    # Unintuitively, pluggy invokes hooks in reverse-registration order.
    # This includes `pytest_confugure` for plugins registered before pytest session starts.
    # See comments on HookCaller._hookimpls in contrib/python/pluggy/py3/pluggy/_hooks.py
    assert pytestconfig._my_conftests == [
        'conftest_non_local/tests/lib_child/conftest.py',
        'conftest_non_local/tests/conftest.py',
        'conftest_non_local/lib2/conftest.py',
        'conftest_non_local/lib/conftest.py',
        'conftest_non_local/conftest.py',
    ]


def test_conftests_are_initial(request):
    assert set(request.session._my_conftests) == {
        'conftest_non_local/conftest.py',
        'conftest_non_local/lib/conftest.py',
        'conftest_non_local/lib2/conftest.py',
        'conftest_non_local/tests/conftest.py',
        'conftest_non_local/tests/lib_child/conftest.py',
    }


def test_conftest_fixture_order(fixture_order):
    assert fixture_order == [
        'conftest_non_local/conftest.py',
        'conftest_non_local/lib/conftest.py',
        'conftest_non_local/lib2/conftest.py',
        'conftest_non_local/tests/conftest.py',
        'conftest_non_local/tests/lib_child/conftest.py',
    ]
