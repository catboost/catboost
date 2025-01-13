from typeguard import check_argument_types, check_return_type, typechecked, typeguard_ignore


@typechecked
def foo(x: int) -> int:
    return x + 1


@typechecked
def bar(x: int) -> int:
    return str(x)  # error: Incompatible return value type (got "str", expected "int")


@typeguard_ignore
def non_typeguard_checked_func(x: int) -> int:
    return str(x)  # error: Incompatible return value type (got "str", expected "int")


def returns_str() -> str:
    return bar(0)  # error: Incompatible return value type (got "int", expected "str")


def arg_type(x: int) -> str:
    return check_argument_types()  # noqa: E501 # error: Incompatible return value type (got "bool", expected "str")


def ret_type() -> str:
    return check_return_type(False)  # noqa: E501 # error: Incompatible return value type (got "bool", expected "str")


_ = arg_type(foo)  # noqa: E501 # error: Argument 1 to "arg_type" has incompatible type "Callable[[int], int]"; expected "int"
_ = foo("typeguard")  # error: Argument 1 to "foo" has incompatible type "str"; expected "int"


@typechecked
class MyClass:
    def __init__(self, x: int = 0) -> None:
        self.x = x

    def add(self, y: int) -> int:
        return self.x + y


def get_value(c: MyClass) -> int:
    return c.x


def create_myclass(x: int) -> MyClass:
    return MyClass(x)


_ = get_value("foo")  # noqa: E501 # error: Argument 1 to "get_value" has incompatible type "str"; expected "MyClass"
_ = MyClass(returns_str())  # noqa: E501 # error: Argument 1 to "MyClass" has incompatible type "str"; expected "int"
