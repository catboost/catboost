import gc
import sys
import traceback
import warnings
from abc import abstractproperty
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial, wraps
from io import BytesIO, StringIO
from typing import (
    AbstractSet, Any, AnyStr, BinaryIO, Callable, Container, Dict, Generator, Generic, Iterable,
    Iterator, List, NamedTuple, Sequence, Set, TextIO, Tuple, Type, TypeVar, Union, TypedDict)
from unittest.mock import MagicMock, Mock

import pytest
from typing_extensions import Literal, NoReturn, Protocol, runtime_checkable

from typeguard import (
    ForwardRefPolicy, TypeChecker, TypeHintWarning, TypeWarning, check_argument_types,
    check_return_type, check_type, function_name, qualified_name, typechecked)

try:
    from typing import Collection
except ImportError:
    # Python 3.6.0+
    Collection = None

try:
    from typing import NewType
except ImportError:
    myint = None
else:
    myint = NewType("myint", int)


TBound = TypeVar('TBound', bound='Parent')
TConstrained = TypeVar('TConstrained', 'Parent', int)
TTypingConstrained = TypeVar('TTypingConstrained', List[int], AbstractSet[str])
TIntStr = TypeVar('TIntStr', int, str)
TIntCollection = TypeVar('TIntCollection', int, Collection)
TParent = TypeVar('TParent', bound='Parent')
TChild = TypeVar('TChild', bound='Child')
T_Foo = TypeVar('T_Foo')
JSONType = Union[str, int, float, bool, None, List['JSONType'], Dict[str, 'JSONType']]

DummyDict = TypedDict('DummyDict', {'x': int}, total=False)
issue_42059 = pytest.mark.xfail(bool(DummyDict.__required_keys__),
                                reason='Fails due to upstream bug BPO-42059')
del DummyDict

Employee = NamedTuple('Employee', [('name', str), ('id', int)])


class FooGeneric(Generic[T_Foo]):
    pass


class Parent:
    pass


class Child(Parent):
    def method(self, a: int):
        pass


class StaticProtocol(Protocol):
    def meth(self) -> None:
        ...


@runtime_checkable
class RuntimeProtocol(Protocol):
    def meth(self) -> None:
        ...


@pytest.fixture(params=[Mock, MagicMock], ids=['mock', 'magicmock'])
def mock_class(request):
    return request.param


@pytest.mark.parametrize('inputval, expected', [
    (qualified_name, 'function'),
    (Child(), '__tests__.test_typeguard.Child'),
    (int, 'int')
], ids=['func', 'instance', 'builtintype'])
def test_qualified_name(inputval, expected):
    assert qualified_name(inputval) == expected


def test_function_name():
    assert function_name(function_name) == 'typeguard.function_name'


def test_check_type_no_memo():
    check_type('foo', [1], List[int])


def test_check_type_bytes():
    pytest.raises(TypeError, check_type, 'foo', 7, bytes).\
        match(r'type of foo must be bytes-like; got int instead')


def test_check_type_no_memo_fail():
    pytest.raises(TypeError, check_type, 'foo', ['a'], List[int]).\
        match(r'type of foo\[0\] must be int; got str instead')


@pytest.mark.parametrize('value', ['bar', b'bar'], ids=['str', 'bytes'])
def test_check_type_anystr(value):
    check_type('foo', value, AnyStr)


def test_check_type_anystr_fail():
    pytest.raises(TypeError, check_type, 'foo', int, AnyStr).\
        match(r'type of foo must match one of the constraints \(bytes, str\); got type instead')


def test_check_return_type():
    def foo() -> int:
        assert check_return_type(0)
        return 0

    foo()


def test_check_return_type_fail():
    def foo() -> int:
        assert check_return_type('foo')
        return 1

    pytest.raises(TypeError, foo).match('type of the return value must be int; got str instead')


def test_check_return_notimplemented():
    class Foo:
        def __eq__(self, other) -> bool:
            assert check_return_type(NotImplemented)
            return NotImplemented

    assert Foo().__eq__(1) is NotImplemented


def test_check_recursive_type():
    check_type('foo', {'a': [1, 2, 3]}, JSONType)
    pytest.raises(TypeError, check_type, 'foo', {'a': (1, 2, 3)}, JSONType, globals=globals()).\
        match(r'type of foo must be one of \(str, int, float, (bool, )?NoneType, '
              r'List\[JSONType\], Dict\[str, JSONType\]\); got dict instead')


def test_exec_no_namespace():
    from textwrap import dedent

    exec(dedent("""
        from typeguard import typechecked

        @typechecked
        def f() -> None:
            pass

        """), {})


class TestCheckArgumentTypes:
    def test_any_type(self):
        def foo(a: Any):
            assert check_argument_types()

        foo('aa')

    def test_mock_value(self, mock_class):
        def foo(a: str, b: int, c: dict, d: Any) -> int:
            assert check_argument_types()

        foo(mock_class(), mock_class(), mock_class(), mock_class())

    def test_callable_exact_arg_count(self):
        def foo(a: Callable[[int, str], int]):
            assert check_argument_types()

        def some_callable(x: int, y: str) -> int:
            pass

        foo(some_callable)

    def test_callable_bad_type(self):
        def foo(a: Callable[..., int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, 5)
        assert str(exc.value) == 'argument "a" must be a callable'

    def test_callable_too_few_arguments(self):
        def foo(a: Callable[[int, str], int]):
            assert check_argument_types()

        def some_callable(x: int) -> int:
            pass

        exc = pytest.raises(TypeError, foo, some_callable)
        assert str(exc.value) == (
            'callable passed as argument "a" has too few arguments in its declaration; expected 2 '
            'but 1 argument(s) declared')

    def test_callable_too_many_arguments(self):
        def foo(a: Callable[[int, str], int]):
            assert check_argument_types()

        def some_callable(x: int, y: str, z: float) -> int:
            pass

        exc = pytest.raises(TypeError, foo, some_callable)
        assert str(exc.value) == (
            'callable passed as argument "a" has too many arguments in its declaration; expected '
            '2 but 3 argument(s) declared')

    def test_callable_mandatory_kwonlyargs(self):
        def foo(a: Callable[[int, str], int]):
            assert check_argument_types()

        def some_callable(x: int, y: str, *, z: float, bar: str) -> int:
            pass

        exc = pytest.raises(TypeError, foo, some_callable)
        assert str(exc.value) == (
            'callable passed as argument "a" has mandatory keyword-only arguments in its '
            'declaration: z, bar')

    def test_callable_class(self):
        """
        Test that passing a class as a callable does not count the "self" argument "a"gainst the
        ones declared in the Callable specification.

        """
        def foo(a: Callable[[int, str], Any]):
            assert check_argument_types()

        class SomeClass:
            def __init__(self, x: int, y: str):
                pass

        foo(SomeClass)

    def test_callable_plain(self):
        def foo(a: Callable):
            assert check_argument_types()

        def callback(a):
            pass

        foo(callback)

    def test_callable_partial_class(self):
        """
        Test that passing a bound method as a callable does not count the "self" argument "a"gainst
        the ones declared in the Callable specification.

        """
        def foo(a: Callable[[int], Any]):
            assert check_argument_types()

        class SomeClass:
            def __init__(self, x: int, y: str):
                pass

        foo(partial(SomeClass, y='foo'))

    def test_callable_bound_method(self):
        """
        Test that passing a bound method as a callable does not count the "self" argument "a"gainst
        the ones declared in the Callable specification.

        """
        def foo(callback: Callable[[int], Any]):
            assert check_argument_types()

        foo(Child().method)

    def test_callable_partial_bound_method(self):
        """
        Test that passing a bound method as a callable does not count the "self" argument "a"gainst
        the ones declared in the Callable specification.

        """
        def foo(callback: Callable[[], Any]):
            assert check_argument_types()

        foo(partial(Child().method, 1))

    def test_callable_defaults(self):
        """
        Test that a callable having "too many" arguments don't raise an error if the extra
        arguments have default values.

        """
        def foo(callback: Callable[[int, str], Any]):
            assert check_argument_types()

        def some_callable(x: int, y: str, z: float = 1.2) -> int:
            pass

        foo(some_callable)

    def test_callable_builtin(self):
        """
        Test that checking a Callable annotation against a builtin callable does not raise an
        error.

        """
        def foo(callback: Callable[[int], Any]):
            assert check_argument_types()

        foo([].append)

    def test_dict_bad_type(self):
        def foo(a: Dict[str, int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, 5)
        assert str(exc.value) == (
            'type of argument "a" must be a dict; got int instead')

    def test_dict_bad_key_type(self):
        def foo(a: Dict[str, int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, {1: 2})
        assert str(exc.value) == 'type of keys of argument "a" must be str; got int instead'

    def test_dict_bad_value_type(self):
        def foo(a: Dict[str, int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, {'x': 'a'})
        assert str(exc.value) == "type of argument \"a\"['x'] must be int; got str instead"

    def test_list_bad_type(self):
        def foo(a: List[int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, 5)
        assert str(exc.value) == (
            'type of argument "a" must be a list; got int instead')

    def test_list_bad_element(self):
        def foo(a: List[int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, [1, 2, 'bb'])
        assert str(exc.value) == (
            'type of argument "a"[2] must be int; got str instead')

    def test_sequence_bad_type(self):
        def foo(a: Sequence[int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, 5)
        assert str(exc.value) == (
            'type of argument "a" must be a sequence; got int instead')

    def test_sequence_bad_element(self):
        def foo(a: Sequence[int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, [1, 2, 'bb'])
        assert str(exc.value) == (
            'type of argument "a"[2] must be int; got str instead')

    def test_abstractset_custom_type(self):
        class DummySet(AbstractSet[int]):
            def __contains__(self, x: object) -> bool:
                return x == 1

            def __len__(self) -> int:
                return 1

            def __iter__(self) -> Iterator[int]:
                yield 1

        def foo(a: AbstractSet[int]):
            assert check_argument_types()

        foo(DummySet())

    def test_abstractset_bad_type(self):
        def foo(a: AbstractSet[int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, 5)
        assert str(exc.value) == 'type of argument "a" must be a set; got int instead'

    def test_set_bad_type(self):
        def foo(a: Set[int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, 5)
        assert str(exc.value) == 'type of argument "a" must be a set; got int instead'

    def test_abstractset_bad_element(self):
        def foo(a: AbstractSet[int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, {1, 2, 'bb'})
        assert str(exc.value) == (
            'type of elements of argument "a" must be int; got str instead')

    def test_set_bad_element(self):
        def foo(a: Set[int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, {1, 2, 'bb'})
        assert str(exc.value) == (
            'type of elements of argument "a" must be int; got str instead')

    def test_tuple_bad_type(self):
        def foo(a: Tuple[int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, 5)
        assert str(exc.value) == (
            'type of argument "a" must be a tuple; got int instead')

    def test_tuple_too_many_elements(self):
        def foo(a: Tuple[int, str]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, (1, 'aa', 2))
        assert str(exc.value) == ('argument "a" has wrong number of elements (expected 2, got 3 '
                                  'instead)')

    def test_tuple_too_few_elements(self):
        def foo(a: Tuple[int, str]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, (1,))
        assert str(exc.value) == ('argument "a" has wrong number of elements (expected 2, got 1 '
                                  'instead)')

    def test_tuple_bad_element(self):
        def foo(a: Tuple[int, str]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, (1, 2))
        assert str(exc.value) == (
            'type of argument "a"[1] must be str; got int instead')

    def test_tuple_ellipsis_bad_element(self):
        def foo(a: Tuple[int, ...]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, (1, 2, 'blah'))
        assert str(exc.value) == (
            'type of argument "a"[2] must be int; got str instead')

    def test_namedtuple(self):
        def foo(bar: Employee):
            assert check_argument_types()

        foo(Employee('bob', 1))

    def test_namedtuple_type_mismatch(self):
        def foo(bar: Employee):
            assert check_argument_types()

        pytest.raises(TypeError, foo, ('bob', 1)).\
            match('type of argument "bar" must be a named tuple of type '
                  r'(__tests__\.test_typeguard\.)?Employee; got tuple instead')

    def test_namedtuple_wrong_field_type(self):
        def foo(bar: Employee):
            assert check_argument_types()

        pytest.raises(TypeError, foo, Employee(2, 1)).\
            match('type of argument "bar".name must be str; got int instead')

    @pytest.mark.parametrize('value', [6, 'aa'])
    def test_union(self, value):
        def foo(a: Union[str, int]):
            assert check_argument_types()

        foo(value)

    def test_union_typing_type(self):
        def foo(a: Union[str, Collection]):
            assert check_argument_types()

        with pytest.raises(TypeError):
            foo(1)

    @pytest.mark.parametrize('value', [6.5, b'aa'])
    def test_union_fail(self, value):
        def foo(a: Union[str, int]):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, value)
        assert str(exc.value) == (
            'type of argument "a" must be one of (str, int); got {} instead'.
            format(value.__class__.__name__))

    @pytest.mark.parametrize('values', [
        (6, 7),
        ('aa', 'bb')
    ], ids=['int', 'str'])
    def test_typevar_constraints(self, values):
        def foo(a: TIntStr, b: TIntStr):
            assert check_argument_types()

        foo(*values)

    @pytest.mark.parametrize('value', [
        [6, 7],
        {'aa', 'bb'}
    ], ids=['int', 'str'])
    def test_typevar_collection_constraints(self, value):
        def foo(a: TTypingConstrained):
            assert check_argument_types()

        foo(value)

    def test_typevar_collection_constraints_fail(self):
        def foo(a: TTypingConstrained):
            assert check_argument_types()

        pytest.raises(TypeError, foo, {1, 2}).\
            match(r'type of argument "a" must match one of the constraints \(List\[int\], '
                  r'AbstractSet\[str\]\); got set instead')

    def test_typevar_constraints_fail(self):
        def foo(a: TIntStr, b: TIntStr):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, 2.5, 'aa')
        assert str(exc.value) == ('type of argument "a" must match one of the constraints '
                                  '(int, str); got float instead')

    def test_typevar_bound(self):
        def foo(a: TParent, b: TParent):
            assert check_argument_types()

        foo(Child(), Child())

    def test_typevar_bound_fail(self):
        def foo(a: TChild, b: TChild):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, Parent(), Parent())
        assert str(exc.value) == ('type of argument "a" must be __tests__.test_typeguard.Child or one of '
                                  'its subclasses; got __tests__.test_typeguard.Parent instead')

    @pytest.mark.skipif(Type is List, reason='typing.Type could not be imported')
    def test_class_bad_subclass(self):
        def foo(a: Type[Child]):
            assert check_argument_types()

        pytest.raises(TypeError, foo, Parent).match(
            '"a" must be a subclass of __tests__.test_typeguard.Child; got __tests__.test_typeguard.Parent instead')

    def test_class_any(self):
        def foo(a: Type[Any]):
            assert check_argument_types()

        foo(str)

    def test_class_union(self):
        def foo(a: Type[Union[str, int]]):
            assert check_argument_types()

        foo(str)
        foo(int)
        pytest.raises(TypeError, foo, tuple).\
            match(r'"a" must match one of the following: \(str, int\); got tuple instead')

    def test_wrapped_function(self):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        @decorator
        def foo(a: 'Child'):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, Parent())
        assert str(exc.value) == ('type of argument "a" must be __tests__.test_typeguard.Child; '
                                  'got __tests__.test_typeguard.Parent instead')

    def test_mismatching_default_type(self):
        def foo(a: str = 1):
            assert check_argument_types()

        pytest.raises(TypeError, foo).match('type of argument "a" must be str; got int instead')

    def test_implicit_default_none(self):
        """
        Test that if the default value is ``None``, a ``None`` argument can be passed.

        """
        def foo(a: str = None):
            assert check_argument_types()

        foo()

    def test_generator(self):
        """Test that argument type checking works in a generator function too."""
        def generate(a: int):
            assert check_argument_types()
            yield a
            yield a + 1

        gen = generate(1)
        next(gen)

    def test_wrapped_generator_no_return_type_annotation(self):
        """Test that return type checking works in a generator function too."""
        @typechecked
        def generate(a: int):
            yield a
            yield a + 1

        gen = generate(1)
        next(gen)

    def test_varargs(self):
        def foo(*args: int):
            assert check_argument_types()

        foo(1, 2)

    def test_varargs_fail(self):
        def foo(*args: int):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, 1, 'a')
        exc.match(r'type of argument "args"\[1\] must be int; got str instead')

    def test_kwargs(self):
        def foo(**kwargs: int):
            assert check_argument_types()

        foo(a=1, b=2)

    def test_kwargs_fail(self):
        def foo(**kwargs: int):
            assert check_argument_types()

        exc = pytest.raises(TypeError, foo, a=1, b='a')
        exc.match(r'type of argument "kwargs"\[\'b\'\] must be int; got str instead')

    def test_generic(self):
        def foo(a: FooGeneric[str]):
            assert check_argument_types()

        foo(FooGeneric[str]())

    @pytest.mark.skipif(myint is None, reason='NewType is not present in the typing module')
    def test_newtype(self):
        def foo(a: myint) -> int:
            assert check_argument_types()
            return 42

        assert foo(1) == 42
        exc = pytest.raises(TypeError, foo, "a")
        assert str(exc.value) == 'type of argument "a" must be int; got str instead'

    @pytest.mark.skipif(Collection is None, reason='typing.Collection is not available')
    def test_collection(self):
        def foo(a: Collection):
            assert check_argument_types()

        pytest.raises(TypeError, foo, True).match(
            'type of argument "a" must be collections.abc.Collection; got bool instead')

    def test_binary_io(self):
        def foo(a: BinaryIO):
            assert check_argument_types()

        foo(BytesIO())

    def test_text_io(self):
        def foo(a: TextIO):
            assert check_argument_types()

        foo(StringIO())

    def test_binary_io_fail(self):
        def foo(a: TextIO):
            assert check_argument_types()

        pytest.raises(TypeError, foo, BytesIO()).match('must be a text based I/O')

    def test_text_io_fail(self):
        def foo(a: BinaryIO):
            assert check_argument_types()

        pytest.raises(TypeError, foo, StringIO()).match('must be a binary I/O')

    def test_binary_io_real_file(self, tmpdir):
        def foo(a: BinaryIO):
            assert check_argument_types()

        with tmpdir.join('testfile').open('wb') as f:
            foo(f)

    def test_text_io_real_file(self, tmpdir):
        def foo(a: TextIO):
            assert check_argument_types()

        with tmpdir.join('testfile').open('w') as f:
            foo(f)

    def test_recursive_type(self):
        def foo(arg: JSONType) -> None:
            assert check_argument_types()

        foo({'a': [1, 2, 3]})
        pytest.raises(TypeError, foo, {'a': (1, 2, 3)}).\
            match(r'type of argument "arg" must be one of \(str, int, float, (bool, )?NoneType, '
                  r'List\[Union\[str, int, float, (bool, )?NoneType, List\[JSONType\], '
                  r'Dict\[str, JSONType\]\]\], '
                  r'Dict\[str, Union\[str, int, float, (bool, )?NoneType, List\[JSONType\], '
                  r'Dict\[str, JSONType\]\]\]\); got dict instead')


class TestTypeChecked:
    def test_typechecked(self):
        @typechecked
        def foo(a: int, b: str) -> str:
            return 'abc'

        assert foo(4, 'abc') == 'abc'

    def test_typechecked_always(self):
        @typechecked(always=True)
        def foo(a: int, b: str) -> str:
            return 'abc'

        assert foo(4, 'abc') == 'abc'

    def test_typechecked_arguments_fail(self):
        @typechecked
        def foo(a: int, b: str) -> str:
            return 'abc'

        exc = pytest.raises(TypeError, foo, 4, 5)
        assert str(exc.value) == 'type of argument "b" must be str; got int instead'

    def test_typechecked_return_type_fail(self):
        @typechecked
        def foo(a: int, b: str) -> str:
            return 6

        exc = pytest.raises(TypeError, foo, 4, 'abc')
        assert str(exc.value) == 'type of the return value must be str; got int instead'

    def test_typechecked_return_typevar_fail(self):
        T = TypeVar('T', int, float)

        @typechecked
        def foo(a: T, b: T) -> T:
            return 'a'

        pytest.raises(TypeError, foo, 4, 2).\
            match(r'type of the return value must match one of the constraints \(int, float\); '
                  r'got str instead')

    def test_typechecked_no_annotations(self, recwarn):
        def foo(a, b):
            pass

        typechecked(foo)

        func_name = function_name(foo)
        assert len(recwarn) == 1
        assert str(recwarn[0].message) == (
            'no type annotations present -- not typechecking {}'.format(func_name))

    def test_return_type_none(self):
        """Check that a declared return type of None is respected."""
        @typechecked
        def foo() -> None:
            return 'a'

        exc = pytest.raises(TypeError, foo)
        assert str(exc.value) == 'type of the return value must be NoneType; got str instead'

    def test_return_type_magicmock(self, mock_class):
        @typechecked
        def foo() -> str:
            return mock_class()

        foo()

    @pytest.mark.parametrize('typehint', [
        Callable[..., int],
        Callable
    ], ids=['parametrized', 'unparametrized'])
    def test_callable(self, typehint):
        @typechecked
        def foo(a: typehint):
            pass

        def some_callable() -> int:
            pass

        foo(some_callable)

    @pytest.mark.parametrize('typehint', [
        List[int],
        List,
        list,
    ], ids=['parametrized', 'unparametrized', 'plain'])
    def test_list(self, typehint):
        @typechecked
        def foo(a: typehint):
            pass

        foo([1, 2])

    @pytest.mark.parametrize('typehint', [
        Dict[str, int],
        Dict,
        dict
    ], ids=['parametrized', 'unparametrized', 'plain'])
    def test_dict(self, typehint):
        @typechecked
        def foo(a: typehint):
            pass

        foo({'x': 2})

    @pytest.mark.parametrize('typehint, value', [
        (Dict, {'x': 2, 6: 4}),
        (List, ['x', 6]),
        (Sequence, ['x', 6]),
        (Set, {'x', 6}),
        (AbstractSet, {'x', 6}),
        (Tuple, ('x', 6)),
    ], ids=['dict', 'list', 'sequence', 'set', 'abstractset', 'tuple'])
    def test_unparametrized_types_mixed_values(self, typehint, value):
        @typechecked
        def foo(a: typehint):
            pass

        foo(value)

    @pytest.mark.parametrize('typehint', [
        Sequence[str],
        Sequence
    ], ids=['parametrized', 'unparametrized'])
    @pytest.mark.parametrize('value', [('a', 'b'), ['a', 'b'], 'abc'],
                             ids=['tuple', 'list', 'str'])
    def test_sequence(self, typehint, value):
        @typechecked
        def foo(a: typehint):
            pass

        foo(value)

    @pytest.mark.parametrize('typehint', [
        Iterable[str],
        Iterable
    ], ids=['parametrized', 'unparametrized'])
    @pytest.mark.parametrize('value', [('a', 'b'), ['a', 'b'], 'abc'],
                             ids=['tuple', 'list', 'str'])
    def test_iterable(self, typehint, value):
        @typechecked
        def foo(a: typehint):
            pass

        foo(value)

    @pytest.mark.parametrize('typehint', [
        Container[str],
        Container
    ], ids=['parametrized', 'unparametrized'])
    @pytest.mark.parametrize('value', [('a', 'b'), ['a', 'b'], 'abc'],
                             ids=['tuple', 'list', 'str'])
    def test_container(self, typehint, value):
        @typechecked
        def foo(a: typehint):
            pass

        foo(value)

    @pytest.mark.parametrize('typehint', [
        AbstractSet[int],
        AbstractSet,
        Set[int],
        Set,
        set
    ], ids=['abstract_parametrized', 'abstract', 'parametrized', 'unparametrized', 'plain'])
    @pytest.mark.parametrize('value', [set(), {6}])
    def test_set(self, typehint, value):
        @typechecked
        def foo(a: typehint):
            pass

        foo(value)

    @pytest.mark.parametrize('typehint', [
        Tuple[int, int],
        Tuple[int, ...],
        Tuple,
        tuple
    ], ids=['parametrized', 'ellipsis', 'unparametrized', 'plain'])
    def test_tuple(self, typehint):
        @typechecked
        def foo(a: typehint):
            pass

        foo((1, 2))

    def test_empty_tuple(self):
        @typechecked
        def foo(a: Tuple[()]):
            pass

        foo(())

    @pytest.mark.skipif(Type is List, reason='typing.Type could not be imported')
    @pytest.mark.parametrize('typehint', [
        Type[Parent],
        Type[TypeVar('UnboundType')],  # noqa: F821
        Type[TypeVar('BoundType', bound=Parent)],  # noqa: F821
        Type,
        type
    ], ids=['parametrized', 'unbound-typevar', 'bound-typevar', 'unparametrized', 'plain'])
    def test_class(self, typehint):
        @typechecked
        def foo(a: typehint):
            pass

        foo(Child)

    @pytest.mark.skipif(Type is List, reason='typing.Type could not be imported')
    def test_class_not_a_class(self):
        @typechecked
        def foo(a: Type[dict]):
            pass

        exc = pytest.raises(TypeError, foo, 1)
        exc.match('type of argument "a" must be a type; got int instead')

    @pytest.mark.parametrize('typehint, value', [
        (complex, complex(1, 5)),
        (complex, 1.0),
        (complex, 1),
        (float, 1.0),
        (float, 1)
    ], ids=['complex-complex', 'complex-float', 'complex-int', 'float-float', 'float-int'])
    def test_numbers(self, typehint, value):
        @typechecked
        def foo(a: typehint):
            pass

        foo(value)

    def test_coroutine_correct_return_type(self):
        @typechecked
        async def foo() -> str:
            return 'foo'

        coro = foo()
        pytest.raises(StopIteration, coro.send, None)

    def test_coroutine_wrong_return_type(self):
        @typechecked
        async def foo() -> str:
            return 1

        coro = foo()
        pytest.raises(TypeError, coro.send, None).\
            match('type of the return value must be str; got int instead')

    def test_bytearray_bytes(self):
        """Test that a bytearray is accepted where bytes are expected."""
        @typechecked
        def foo(x: bytes) -> None:
            pass

        foo(bytearray([1]))

    def test_bytearray_memoryview(self):
        """Test that a bytearray is accepted where bytes are expected."""
        @typechecked
        def foo(x: bytes) -> None:
            pass

        foo(memoryview(b'foo'))

    def test_class_decorator(self):
        @typechecked
        class Foo:
            @staticmethod
            def staticmethod() -> int:
                return 'foo'

            @classmethod
            def classmethod(cls) -> int:
                return 'foo'

            def method(self) -> int:
                return 'foo'

            @property
            def prop(self) -> int:
                return 'foo'

            @property
            def prop2(self) -> int:
                return 'foo'

            @prop2.setter
            def prop2(self, value: int) -> None:
                pass

        pattern = 'type of the return value must be int; got str instead'
        pytest.raises(TypeError, Foo.staticmethod).match(pattern)
        pytest.raises(TypeError, Foo.classmethod).match(pattern)
        pytest.raises(TypeError, Foo().method).match(pattern)

        with pytest.raises(TypeError) as raises:
            Foo().prop

        assert raises.value.args[0] == pattern

        with pytest.raises(TypeError) as raises:
            Foo().prop2

        assert raises.value.args[0] == pattern

        with pytest.raises(TypeError) as raises:
            Foo().prop2 = 'foo'

        assert raises.value.args[0] == 'type of argument "value" must be int; got str instead'

    @pytest.mark.parametrize('annotation', [
        Generator[int, str, List[str]],
        Generator,
        Iterable[int],
        Iterable,
        Iterator[int],
        Iterator
    ], ids=['generator', 'bare_generator', 'iterable', 'bare_iterable', 'iterator',
            'bare_iterator'])
    def test_generator(self, annotation):
        @typechecked
        def genfunc() -> annotation:
            val1 = yield 2
            val2 = yield 3
            val3 = yield 4
            return [val1, val2, val3]

        gen = genfunc()
        with pytest.raises(StopIteration) as exc:
            value = next(gen)
            while True:
                value = gen.send(str(value))
                assert isinstance(value, int)

        assert exc.value.value == ['2', '3', '4']

    @pytest.mark.parametrize('annotation', [
        Generator[int, str, None],
        Iterable[int],
        Iterator[int]
    ], ids=['generator', 'iterable', 'iterator'])
    def test_generator_bad_yield(self, annotation):
        @typechecked
        def genfunc() -> annotation:
            yield 'foo'

        gen = genfunc()
        with pytest.raises(TypeError) as exc:
            next(gen)

        exc.match('type of value yielded from generator must be int; got str instead')

    def test_generator_bad_send(self):
        @typechecked
        def genfunc() -> Generator[int, str, None]:
            yield 1
            yield 2

        gen = genfunc()
        next(gen)
        with pytest.raises(TypeError) as exc:
            gen.send(2)

        exc.match('type of value sent to generator must be str; got int instead')

    def test_generator_bad_return(self):
        @typechecked
        def genfunc() -> Generator[int, str, str]:
            yield 1
            return 6

        gen = genfunc()
        next(gen)
        with pytest.raises(TypeError) as exc:
            gen.send('foo')

        exc.match('type of return value must be str; got int instead')

    def test_return_generator(self):
        @typechecked
        def genfunc() -> Generator[int, None, None]:
            yield 1

        @typechecked
        def foo() -> Generator[int, None, None]:
            return genfunc()

        foo()

    def test_builtin_decorator(self):
        @typechecked
        @lru_cache()
        def func(x: int) -> None:
            pass

        func(3)
        func(3)
        pytest.raises(TypeError, func, 'foo').\
            match('type of argument "x" must be int; got str instead')

        # Make sure that @lru_cache is still being used
        cache_info = func.__wrapped__.cache_info()
        assert cache_info.hits == 1

    def test_local_class(self):
        @typechecked
        class LocalClass:
            class Inner:
                pass

            def create_inner(self) -> 'Inner':
                return self.Inner()

        retval = LocalClass().create_inner()
        assert isinstance(retval, LocalClass.Inner)

    def test_local_class_async(self):
        @typechecked
        class LocalClass:
            class Inner:
                pass

            async def create_inner(self) -> 'Inner':
                return self.Inner()

        coro = LocalClass().create_inner()
        exc = pytest.raises(StopIteration, coro.send, None)
        retval = exc.value.value
        assert isinstance(retval, LocalClass.Inner)

    def test_callable_nonmember(self):
        class CallableClass:
            def __call__(self):
                pass

        @typechecked
        class LocalClass:
            some_callable = CallableClass()

    def test_inherited_class_method(self):
        @typechecked
        class Parent:
            @classmethod
            def foo(cls, x: str) -> str:
                return cls.__name__

        @typechecked
        class Child(Parent):
            pass

        assert Child.foo('bar') == 'Child'
        pytest.raises(TypeError, Child.foo, 1)

    def test_class_property(self):
        @typechecked
        class Foo:
            def __init__(self) -> None:
                self.foo = 'foo'

            @property
            def prop(self) -> int:
                """My property."""
                return 4

            @property
            def prop2(self) -> str:
                return self.foo

            @prop2.setter
            def prop2(self, value: str) -> None:
                self.foo = value

        assert Foo.__dict__["prop"].__doc__.strip() == "My property."
        f = Foo()
        assert f.prop == 4
        assert f.prop2 == 'foo'
        f.prop2 = 'bar'
        assert f.prop2 == 'bar'

        with pytest.raises(TypeError) as raises:
            f.prop2 = 3

        assert raises.value.args[0] == 'type of argument "value" must be str; got int instead'

    def test_decorator_factory_no_annotations(self):
        class CallableClass:
            def __call__(self):
                pass

        def decorator_factory():
            def decorator(f):
                cmd = CallableClass()
                return cmd

            return decorator

        with pytest.warns(UserWarning):
            @typechecked
            @decorator_factory()
            def foo():
                pass

    @pytest.mark.skipif(sys.version_info >= (3, 12), reason="Fail wint Python 3.12")
    @pytest.mark.parametrize('annotation', [TBound, TConstrained], ids=['bound', 'constrained'])
    def test_typevar_forwardref(self, annotation):
        @typechecked
        def func(x: annotation) -> None:
            pass

        func(Parent())
        func(Child())
        pytest.raises(TypeError, func, 'foo')

    @pytest.mark.parametrize('protocol_cls', [RuntimeProtocol, StaticProtocol])
    def test_protocol(self, protocol_cls):
        @typechecked
        def foo(arg: protocol_cls) -> None:
            pass

        class Foo:
            def meth(self) -> None:
                pass

        foo(Foo())

    def test_protocol_fail(self):
        @typechecked
        def foo(arg: RuntimeProtocol) -> None:
            pass

        pytest.raises(TypeError, foo, object()).\
            match(r'type of argument "arg" \(object\) is not compatible with the RuntimeProtocol '
                  'protocol')

    def test_noreturn(self):
        @typechecked
        def foo() -> NoReturn:
            pass

        pytest.raises(TypeError, foo).match(r'foo\(\) was declared never to return but it did')

    def test_recursive_type(self):
        @typechecked
        def foo(arg: JSONType) -> None:
            pass

        foo({'a': [1, 2, 3]})
        pytest.raises(TypeError, foo, {'a': (1, 2, 3)}).\
            match(r'type of argument "arg" must be one of \(str, int, float, (bool, )?NoneType, '
                  r'List\[Union\[str, int, float, (bool, )?NoneType, List\[JSONType\], '
                  r'Dict\[str, JSONType\]\]\], '
                  r'Dict\[str, Union\[str, int, float, (bool, )?NoneType, List\[JSONType\], '
                  r'Dict\[str, JSONType\]\]\]\); got dict instead')

    def test_literal(self):
        from http import HTTPStatus

        @typechecked
        def foo(a: Literal[1, True, 'x', b'y', HTTPStatus.ACCEPTED]):
            pass

        foo(HTTPStatus.ACCEPTED)
        pytest.raises(TypeError, foo, 4).match(r"must be one of \(1, True, 'x', b'y', "
                                               r"<HTTPStatus.ACCEPTED: 202>\); got 4 instead$")

    def test_literal_union(self):
        @typechecked
        def foo(a: Union[str, Literal[1, 6, 8]]):
            pass

        foo(6)
        pytest.raises(TypeError, foo, 4).\
            match(r'must be one of \(str, Literal\[1, 6, 8\]\); got int instead$')

    def test_literal_nested(self):
        @typechecked
        def foo(a: Literal[1, Literal['x', 'a', Literal['z']], 6, 8]):
            pass

        foo('z')
        pytest.raises(TypeError, foo, 4).match(r"must be one of \(1, 'x', 'a', 'z', 6, 8\); "
                                               r"got 4 instead$")

    def test_literal_illegal_value(self):
        @typechecked
        def foo(a: Literal[1, 1.1]):
            pass

        pytest.raises(TypeError, foo, 4).match(r"Illegal literal value: 1.1$")

    @pytest.mark.parametrize('value, total, error_re', [
        pytest.param({'x': 6, 'y': 'foo'}, True, None, id='correct'),
        pytest.param({'y': 'foo'}, True, r'required key\(s\) \("x"\) missing from argument "arg"',
                     id='missing_x'),
        pytest.param({'x': 6, 'y': 3}, True,
                     'type of dict item "y" for argument "arg" must be str; got int instead',
                     id='wrong_y'),
        pytest.param({'x': 6}, True, r'required key\(s\) \("y"\) missing from argument "arg"',
                     id='missing_y_error'),
        pytest.param({'x': 6}, False, None, id='missing_y_ok', marks=[issue_42059]),
        pytest.param({'x': 'abc'}, False,
                     'type of dict item "x" for argument "arg" must be int; got str instead',
                     id='wrong_x', marks=[issue_42059]),
        pytest.param({'x': 6, 'foo': 'abc'}, False, r'extra key\(s\) \("foo"\) in argument "arg"',
                     id='unknown_key')
    ])
    def test_typed_dict(self, value, total, error_re):
        DummyDict = TypedDict('DummyDict', {'x': int, 'y': str}, total=total)

        @typechecked
        def foo(arg: DummyDict):
            pass

        if error_re:
            pytest.raises(TypeError, foo, value).match(error_re)
        else:
            foo(value)

    def test_class_abstract_property(self):
        """Regression test for #206."""

        @typechecked
        class Foo:
            @abstractproperty
            def dummyproperty(self):
                pass

        assert isinstance(Foo.dummyproperty, abstractproperty)


class TestTypeChecker:
    @pytest.fixture
    def executor(self):
        executor = ThreadPoolExecutor(1)
        yield executor
        executor.shutdown()

    @pytest.fixture
    def checker(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            return TypeChecker(__name__)

    @staticmethod
    def generatorfunc() -> Generator[int, None, None]:
        yield 1

    @staticmethod
    def bad_generatorfunc() -> Generator[int, None, None]:
        yield 1
        yield 'foo'

    @staticmethod
    def error_function() -> float:
        return 1 / 0

    def test_check_call_args(self, checker: TypeChecker):
        def foo(a: int):
            pass

        with checker, pytest.warns(TypeWarning) as record:
            assert checker.active
            foo(1)
            foo('x')

        assert not checker.active
        foo('x')

        assert len(record) == 1
        warning = record[0].message
        assert warning.error == 'type of argument "a" must be int; got str instead'
        assert warning.func is foo
        assert isinstance(warning.stack, list)
        buffer = StringIO()
        warning.print_stack(buffer)
        assert len(buffer.getvalue()) > 100

    def test_check_return_value(self, checker: TypeChecker):
        def foo() -> int:
            return 'x'

        with checker, pytest.warns(TypeWarning) as record:
            foo()

        assert len(record) == 1
        assert record[0].message.error == 'type of the return value must be int; got str instead'

    def test_threaded_check_call_args(self, checker: TypeChecker, executor):
        def foo(a: int):
            pass

        with checker, pytest.warns(TypeWarning) as record:
            executor.submit(foo, 1).result()
            executor.submit(foo, 'x').result()

        executor.submit(foo, 'x').result()

        assert len(record) == 1
        warning = record[0].message
        assert warning.error == 'type of argument "a" must be int; got str instead'
        assert warning.func is foo

    def test_double_start(self, checker: TypeChecker):
        """Test that the same type checker can't be started twice while running."""
        with checker:
            pytest.raises(RuntimeError, checker.start).match('type checker already running')

    def test_nested(self):
        """Test that nesting of type checker context managers works as expected."""
        def foo(a: int):
            pass

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore', DeprecationWarning)
            parent = TypeChecker(__name__)
            child = TypeChecker(__name__)

        with parent, pytest.warns(TypeWarning) as record:
            foo('x')
            with child:
                foo('x')

        assert len(record) == 3

    def test_existing_profiler(self, checker: TypeChecker):
        """
        Test that an existing profiler function is chained with the type checker and restored after
        the block is exited.

        """
        def foo(a: int):
            pass

        def profiler(frame, event, arg):
            nonlocal profiler_run_count
            if event in ('call', 'return'):
                profiler_run_count += 1

            if old_profiler:
                old_profiler(frame, event, arg)

        profiler_run_count = 0
        old_profiler = sys.getprofile()
        sys.setprofile(profiler)
        try:
            with checker, pytest.warns(TypeWarning) as record:
                foo(1)
                foo('x')

            assert sys.getprofile() is profiler
        finally:
            sys.setprofile(old_profiler)

        assert profiler_run_count
        assert len(record) == 1

    def test_generator(self, checker):
        with checker, pytest.warns(None) as record:
            gen = self.generatorfunc()
            assert next(gen) == 1

        assert len(record) == 0

    def test_generator_wrong_yield(self, checker):
        with checker, pytest.warns(TypeWarning) as record:
            gen = self.bad_generatorfunc()
            assert list(gen) == [1, 'foo']

        assert len(record) == 1
        assert 'type of yielded value must be int; got str instead' in str(record[0].message)

    def test_exception(self, checker):
        with checker, pytest.warns(None) as record:
            pytest.raises(ZeroDivisionError, self.error_function)

        assert len(record) == 0

    @pytest.mark.parametrize('policy', [ForwardRefPolicy.WARN, ForwardRefPolicy.GUESS],
                             ids=['warn', 'guess'])
    def test_forward_ref_policy_resolution_fails(self, checker, policy):
        def unresolvable_annotation(x: 'OrderedDict'):  # noqa
            pass

        checker.annotation_policy = policy
        gc.collect()  # prevent find_function() from finding more than one instance of the function
        with checker, pytest.warns(TypeHintWarning) as record:
            unresolvable_annotation({})

        assert len(record) == 1
        assert ("unresolvable_annotation: name 'OrderedDict' is not defined"
                in str(record[0].message))
        assert 'x' not in unresolvable_annotation.__annotations__

    def test_forward_ref_policy_guess(self, checker):
        import collections

        def unresolvable_annotation(x: 'OrderedDict'):  # noqa
            pass

        checker.annotation_policy = ForwardRefPolicy.GUESS
        with checker, pytest.warns(TypeHintWarning) as record:
            unresolvable_annotation(collections.OrderedDict())

        assert len(record) == 1
        assert str(record[0].message).startswith("Replaced forward declaration 'OrderedDict' in")
        assert unresolvable_annotation.__annotations__['x'] is collections.OrderedDict


class TestTracebacks:
    def test_short_tracebacks(self):
        def foo(a: Callable[..., int]):
            assert check_argument_types()

        try:
            foo(1)
        except TypeError:
            _, _, tb = sys.exc_info()
            parts = traceback.extract_tb(tb)
            typeguard_lines = [part for part in parts
                               if part.filename.endswith("typeguard/__init__.py")]
            assert len(typeguard_lines) == 1
