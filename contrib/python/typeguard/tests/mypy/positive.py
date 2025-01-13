from typing import Callable

from typeguard import check_argument_types, check_return_type, typechecked


@typechecked
def foo(x: str) -> str:
    return "hello " + x


def takes_callable(f: Callable[[str], str]) -> str:
    return f("typeguard")


takes_callable(foo)


def has_valid_arguments(x: int, y: str) -> bool:
    return check_argument_types()


def has_valid_return_type(y: str) -> str:
    check_return_type(y)
    return y


@typechecked
class MyClass:

    def __init__(self, x: int) -> None:
        self.x = x

    def add(self, y: int) -> int:
        return self.x + y


def get_value(c: MyClass) -> int:
    return c.x


@typechecked
def get_value_checked(c: MyClass) -> int:
    return c.x


def create_myclass(x: int) -> MyClass:
    return MyClass(x)


@typechecked
def create_myclass_checked(x: int) -> MyClass:
    return MyClass(x)


get_value(create_myclass(3))
get_value_checked(create_myclass_checked(1))
