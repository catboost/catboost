import pytest
import sys

from library.python.pytest.yatest_tools import split_node_id


@pytest.fixture(params=["", "[param1,param2]"])
def parameters(request):
    return request.param


@pytest.mark.parametrize("node_id,expected_class_name,expected_test_name",
    (
        ("package.test_script.py::test_name", "package.test_script.py", "test_name"),
        ("package.test_script.py", "package.test_script.py", "package.test_script.py"),
        ("package.test_script.py::class_name::test_name", "package.test_script.py::class_name", "test_name"),
        ("package.test_script.py::class_name::subclass_name::test_name", "package.test_script.py::class_name", "test_name"),
    )
)
def test_split_node_id_without_path(parameters, node_id, expected_class_name, expected_test_name):
    got = split_node_id(node_id + parameters)
    assert (expected_class_name, expected_test_name + parameters) == got


@pytest.mark.parametrize("node_id,expected_class_name,expected_test_name",
    (
        ("/arcadia/root/package/test_script.py", "package.test_script.py", "package.test_script.py"),
        ("/arcadia/root/package/test_script.py::test_name","package.test_script.py", "test_name"),
        ("/arcadia/root/package/test_script.py::class_name::test_name", "package.test_script.py::class_name", "test_name"),
        ("/arcadia/root/package/test_script.py::class_name::subclass_name::test_name", "package.test_script.py::class_name", "test_name"),
        # If module is not found in sys.extra_modules use basename as a class name
        ("/arcadia/root/package/test_script2.py::test_name", "test_script2.py", "test_name"),
    )
)
def test_split_node_id_with_path(mocker, parameters, node_id, expected_class_name, expected_test_name):
    mocker.patch.object(sys, 'extra_modules', sys.extra_modules | {'__tests__.package.test_script'})
    got = split_node_id(node_id + parameters)
    assert (expected_class_name, expected_test_name + parameters) == got


@pytest.mark.parametrize("node_id,expected_class_name,expected_test_name",
    (
        ("package.test_script.py::test_name", "package.test_script.py", "test_suffix"),
        ("package.test_script.py", "package.test_script.py", "test_suffix"),
        ("package.test_script.py::class_name::test_name", "package.test_script.py", "test_suffix"),
        ("package.test_script.py::class_name::subclass_name::test_name", "package.test_script.py", "test_suffix"),
    )
)
def test_split_node_id_with_test_suffix(mocker, parameters, node_id, expected_class_name, expected_test_name):
    got = split_node_id(node_id + parameters, "test_suffix")
    assert (expected_class_name, expected_test_name + parameters) == got
