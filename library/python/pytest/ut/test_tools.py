import pytest
import sys

from library.python.pytest import config
from library.python.pytest import yatest_tools


config.set_test_mode()


@pytest.fixture(params=["", "[param1,param2]"])
def parameters(request):
    return request.param


@pytest.mark.parametrize(
    "node_id,expected_class_name,expected_test_name",
    (
        ("package.test_script.py::test_name", "package.test_script.py", "test_name"),
        ("package.test_script.py", "package.test_script.py", "package.test_script.py"),
        ("package.test_script.py::class_name::test_name", "package.test_script.py::class_name", "test_name"),
        (
            "package.test_script.py::class_name::subclass_name::test_name",
            "package.test_script.py::class_name",
            "test_name",
        ),
    ),
)
def test_split_node_id_without_path(parameters, node_id, expected_class_name, expected_test_name):
    got = yatest_tools.split_node_id(node_id + parameters)
    assert (expected_class_name, expected_test_name + parameters) == got


@pytest.mark.parametrize(
    "node_id,expected_class_name,expected_test_name",
    (
        ("/arcadia/root/package/test_script.py", "package.test_script.py", "package.test_script.py"),
        ("/arcadia/root/package/test_script.py::test_name", "package.test_script.py", "test_name"),
        (
            "/arcadia/root/package/test_script.py::class_name::test_name",
            "package.test_script.py::class_name",
            "test_name",
        ),
        (
            "/arcadia/root/package/test_script.py::class_name::subclass_name::test_name",
            "package.test_script.py::class_name",
            "test_name",
        ),
    ),
)
def test_split_node_id_with_path(mocker, parameters, node_id, expected_class_name, expected_test_name):
    mocker.patch.object(sys, 'extra_modules', sys.extra_modules | {'__tests__.package.test_script'})
    got = yatest_tools.split_node_id(node_id + parameters)
    assert (expected_class_name, expected_test_name + parameters) == got


def test_missing_module(parameters):
    # An errno must be raised if module is not found in sys.extra_modules
    with pytest.raises(yatest_tools.MissingTestModule):
        yatest_tools.split_node_id("/arcadia/root/package/test_script2.py::test_name" + parameters)


@pytest.mark.parametrize(
    "node_id,expected_class_name,expected_test_name",
    (
        ("package.test_script.py::test_name", "package.test_script.py", "test_suffix"),
        ("package.test_script.py", "package.test_script.py", "test_suffix"),
        ("package.test_script.py::class_name::test_name", "package.test_script.py", "test_suffix"),
        ("package.test_script.py::class_name::subclass_name::test_name", "package.test_script.py", "test_suffix"),
    ),
)
def test_split_node_id_with_test_suffix(parameters, node_id, expected_class_name, expected_test_name):
    got = yatest_tools.split_node_id(node_id + parameters, "test_suffix")
    assert (expected_class_name, expected_test_name + parameters) == got


@pytest.mark.parametrize(
    "node_id,expected_class_name,expected_test_name",
    [
        ("/arcadia/data/b/a/test.py::test_b_a", "b.a.test.py", "test_b_a"),
        ("/arcadia/data/a/test.py::test_a", "a.test.py", "test_a"),
        ("/arcadia/data/test.py::test", "test.py", "test"),
        ("b/a/test.py::test_b_a", "b.a.test.py", "test_b_a"),
    ],
)
def test_path_resolving_for_local_conftest_load_policy(
    mocker, parameters, node_id, expected_class_name, expected_test_name
):
    # Order matters
    extra_modules = [
        '__tests__.b.a.test',
        '__tests__.test',
        '__tests__.a.test',
    ]
    mocker.patch.object(sys, 'extra_modules', extra_modules)
    got = yatest_tools.split_node_id(node_id + parameters)
    assert (expected_class_name, expected_test_name + parameters) == got


DATA = [
    (
        "simple_bt",
        """
path/test.py:15: in test
    return foo(0)
path/test.py:5: in foo
    b, a = 1, 1 / p
E   ZeroDivisionError: integer division or modulo by zero
""",
    ),
    (
        "nested_by",
        """
path/test.py:13: in test
    raise Exception("123\n123\npath/test.py:15: in test")
E   Exception: 123
E   123
E   path/test.py:15: in test

During handling of the above exception, another exception occurred:
path/test.py:15: in test
    return foo(0)
path/test.py:5: in foo
    b, a = 1, 1 / p
E   ZeroDivisionError: division by zero
""",
    ),
]


@pytest.mark.parametrize("bt", [x[1] for x in DATA], ids=([x[0] for x in DATA]))
def test_colorize_pytest_error(bt):
    return yatest_tools.colorize_pytest_error(bt)
