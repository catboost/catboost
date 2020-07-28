import enum
import pytest

from yatest_lib import external


class MyEnum(enum.Enum):
    VAL1 = 1
    VAL2 = 2


@pytest.mark.parametrize("data, expected_val, expected_type", [
    ({}, {}, dict),
    (MyEnum.VAL1, "MyEnum.VAL1", str),
    ({MyEnum.VAL1: MyEnum.VAL2}, {"MyEnum.VAL1": "MyEnum.VAL2"}, dict),
])
def test_serialize(data, expected_val, expected_type):
    data = external.serialize(data)
    assert expected_type == type(data), data
    assert expected_val == data
