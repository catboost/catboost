import sys
import os
import contextlib
import collections
import subprocess
from unittest import TestCase, main

from typing_extensions import Annotated, NoReturn, ClassVar, IntVar
from typing_extensions import ContextManager, Counter, Deque, DefaultDict
from typing_extensions import NewType, TypeAlias, overload
from typing import Dict, List
import typing
import typing_extensions


T = typing.TypeVar('T')
KT = typing.TypeVar('KT')
VT = typing.TypeVar('VT')


class BaseTestCase(TestCase):

    def assertIsSubclass(self, cls, class_or_tuple, msg=None):
        if not issubclass(cls, class_or_tuple):
            message = '%r is not a subclass of %r' % (cls, class_or_tuple)
            if msg is not None:
                message += ' : %s' % msg
            raise self.failureException(message)

    def assertNotIsSubclass(self, cls, class_or_tuple, msg=None):
        if issubclass(cls, class_or_tuple):
            message = '%r is a subclass of %r' % (cls, class_or_tuple)
            if msg is not None:
                message += ' : %s' % msg
            raise self.failureException(message)


class Employee(object):
    pass


class NoReturnTests(BaseTestCase):

    def test_noreturn_instance_type_error(self):
        with self.assertRaises(TypeError):
            isinstance(42, NoReturn)

    def test_noreturn_subclass_type_error(self):
        with self.assertRaises(TypeError):
            issubclass(Employee, NoReturn)
        with self.assertRaises(TypeError):
            issubclass(NoReturn, Employee)

    def test_repr(self):
        if hasattr(typing, 'NoReturn'):
            self.assertEqual(repr(NoReturn), 'typing.NoReturn')
        else:
            self.assertEqual(repr(NoReturn), 'typing_extensions.NoReturn')

    def test_not_generic(self):
        with self.assertRaises(TypeError):
            NoReturn[int]

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class A(NoReturn):
                pass
        with self.assertRaises(TypeError):
            class A(type(NoReturn)):
                pass

    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            NoReturn()
        with self.assertRaises(TypeError):
            type(NoReturn)()


class ClassVarTests(BaseTestCase):

    def test_basics(self):
        with self.assertRaises(TypeError):
            ClassVar[1]
        with self.assertRaises(TypeError):
            ClassVar[int, str]
        with self.assertRaises(TypeError):
            ClassVar[int][str]

    def test_repr(self):
        self.assertEqual(repr(ClassVar), 'typing.ClassVar')
        cv = ClassVar[int]
        self.assertEqual(repr(cv), 'typing.ClassVar[int]')
        cv = ClassVar[Employee]
        self.assertEqual(repr(cv), 'typing.ClassVar[%s.Employee]' % __name__)

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(ClassVar)):
                pass
        with self.assertRaises(TypeError):
            class C(type(ClassVar[int])):
                pass

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            ClassVar()
        with self.assertRaises(TypeError):
            type(ClassVar)()
        with self.assertRaises(TypeError):
            type(ClassVar[typing.Optional[int]])()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, ClassVar[int])
        with self.assertRaises(TypeError):
            issubclass(int, ClassVar)


class IntVarTests(BaseTestCase):
    def test_valid(self):
        T_ints = IntVar("T_ints")  # noqa

    def test_invalid(self):
        with self.assertRaises(TypeError):
            T_ints = IntVar("T_ints", int)
        with self.assertRaises(TypeError):
            T_ints = IntVar("T_ints", bound=int)
        with self.assertRaises(TypeError):
            T_ints = IntVar("T_ints", covariant=True)  # noqa


class CollectionsAbcTests(BaseTestCase):

    def test_isinstance_collections(self):
        self.assertNotIsInstance(1, collections.Mapping)
        self.assertNotIsInstance(1, collections.Iterable)
        self.assertNotIsInstance(1, collections.Container)
        self.assertNotIsInstance(1, collections.Sized)
        with self.assertRaises(TypeError):
            isinstance(collections.deque(), typing_extensions.Deque[int])
        with self.assertRaises(TypeError):
            issubclass(collections.Counter, typing_extensions.Counter[str])

    def test_contextmanager(self):
        @contextlib.contextmanager
        def manager():
            yield 42

        cm = manager()
        self.assertIsInstance(cm, ContextManager)
        self.assertNotIsInstance(42, ContextManager)

        with self.assertRaises(TypeError):
            isinstance(42, ContextManager[int])
        with self.assertRaises(TypeError):
            isinstance(cm, ContextManager[int])
        with self.assertRaises(TypeError):
            issubclass(type(cm), ContextManager[int])

    def test_counter(self):
        self.assertIsSubclass(collections.Counter, Counter)
        self.assertIs(type(Counter()), collections.Counter)
        self.assertIs(type(Counter[T]()), collections.Counter)
        self.assertIs(type(Counter[int]()), collections.Counter)

        class A(Counter[int]): pass
        class B(Counter[T]): pass

        self.assertIsInstance(A(), collections.Counter)
        self.assertIs(type(B[int]()), B)
        self.assertEqual(B.__bases__, (typing_extensions.Counter,))

    def test_deque(self):
        self.assertIsSubclass(collections.deque, Deque)
        self.assertIs(type(Deque()), collections.deque)
        self.assertIs(type(Deque[T]()), collections.deque)
        self.assertIs(type(Deque[int]()), collections.deque)

        class A(Deque[int]): pass
        class B(Deque[T]): pass

        self.assertIsInstance(A(), collections.deque)
        self.assertIs(type(B[int]()), B)

    def test_defaultdict_instantiation(self):
        self.assertIsSubclass(collections.defaultdict, DefaultDict)
        self.assertIs(type(DefaultDict()), collections.defaultdict)
        self.assertIs(type(DefaultDict[KT, VT]()), collections.defaultdict)
        self.assertIs(type(DefaultDict[str, int]()), collections.defaultdict)

        class A(DefaultDict[str, int]): pass
        class B(DefaultDict[KT, VT]): pass

        self.assertIsInstance(A(), collections.defaultdict)
        self.assertIs(type(B[str, int]()), B)


class NewTypeTests(BaseTestCase):

    def test_basic(self):
        UserId = NewType('UserId', int)
        UserName = NewType('UserName', str)
        self.assertIsInstance(UserId(5), int)
        self.assertIsInstance(UserName('Joe'), type('Joe'))
        self.assertEqual(UserId(5) + 1, 6)

    def test_errors(self):
        UserId = NewType('UserId', int)
        UserName = NewType('UserName', str)
        with self.assertRaises(TypeError):
            issubclass(UserId, int)
        with self.assertRaises(TypeError):
            class D(UserName):
                pass


class OverloadTests(BaseTestCase):

    def test_overload_fails(self):
        with self.assertRaises(RuntimeError):
            @overload
            def blah():
                pass

            blah()

    def test_overload_succeeds(self):
        @overload
        def blah():
            pass

        def blah():
            pass

        blah()


class AnnotatedTests(BaseTestCase):

    def test_repr(self):
        self.assertEqual(
            repr(Annotated[int, 4, 5]),
            "typing_extensions.Annotated[int, 4, 5]"
        )
        self.assertEqual(
            repr(Annotated[List[int], 4, 5]),
            "typing_extensions.Annotated[typing.List[int], 4, 5]"
        )
        self.assertEqual(repr(Annotated), "typing_extensions.Annotated")

    def test_flatten(self):
        A = Annotated[Annotated[int, 4], 5]
        self.assertEqual(A, Annotated[int, 4, 5])
        self.assertEqual(A.__metadata__, (4, 5))

    def test_specialize(self):
        L = Annotated[List[T], "my decoration"]
        LI = Annotated[List[int], "my decoration"]
        self.assertEqual(L[int], Annotated[List[int], "my decoration"])
        self.assertEqual(L[int].__metadata__, ("my decoration",))
        with self.assertRaises(TypeError):
            LI[int]
        with self.assertRaises(TypeError):
            L[int, float]

    def test_hash_eq(self):
        self.assertEqual(len({Annotated[int, 4, 5], Annotated[int, 4, 5]}), 1)
        self.assertNotEqual(Annotated[int, 4, 5], Annotated[int, 5, 4])
        self.assertNotEqual(Annotated[int, 4, 5], Annotated[str, 4, 5])
        self.assertNotEqual(Annotated[int, 4], Annotated[int, 4, 4])
        self.assertEqual(
            {Annotated[int, 4, 5], Annotated[int, 4, 5], Annotated[T, 4, 5]},
            {Annotated[int, 4, 5], Annotated[T, 4, 5]}
        )

    def test_instantiate(self):
        class C:
            classvar = 4

            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                if not isinstance(other, C):
                    return NotImplemented
                return other.x == self.x

        A = Annotated[C, "a decoration"]
        a = A(5)
        c = C(5)
        self.assertEqual(a, c)
        self.assertEqual(a.x, c.x)
        self.assertEqual(a.classvar, c.classvar)

    def test_instantiate_generic(self):
        MyCount = Annotated[typing_extensions.Counter[T], "my decoration"]
        self.assertEqual(MyCount([4, 4, 5]), {4: 2, 5: 1})
        self.assertEqual(MyCount[int]([4, 4, 5]), {4: 2, 5: 1})

    def test_cannot_instantiate_forward(self):
        A = Annotated["int", (5, 6)]
        with self.assertRaises(TypeError):
            A(5)

    def test_cannot_instantiate_type_var(self):
        A = Annotated[T, (5, 6)]
        with self.assertRaises(TypeError):
            A(5)

    def test_cannot_getattr_typevar(self):
        with self.assertRaises(AttributeError):
            Annotated[T, (5, 7)].x

    def test_attr_passthrough(self):
        class C:
            classvar = 4

        A = Annotated[C, "a decoration"]
        self.assertEqual(A.classvar, 4)
        A.x = 5
        self.assertEqual(C.x, 5)

    def test_hash_eq(self):
        self.assertEqual(len({Annotated[int, 4, 5], Annotated[int, 4, 5]}), 1)
        self.assertNotEqual(Annotated[int, 4, 5], Annotated[int, 5, 4])
        self.assertNotEqual(Annotated[int, 4, 5], Annotated[str, 4, 5])
        self.assertNotEqual(Annotated[int, 4], Annotated[int, 4, 4])
        self.assertEqual(
            {Annotated[int, 4, 5], Annotated[int, 4, 5], Annotated[T, 4, 5]},
            {Annotated[int, 4, 5], Annotated[T, 4, 5]}
        )

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(Annotated):
                pass

    def test_cannot_check_instance(self):
        with self.assertRaises(TypeError):
            isinstance(5, Annotated[int, "positive"])

    def test_cannot_check_subclass(self):
        with self.assertRaises(TypeError):
            issubclass(int, Annotated[int, "positive"])

    def test_subst(self):
        dec = "a decoration"
        dec2 = "another decoration"

        S = Annotated[T, dec2]
        self.assertEqual(S[int], Annotated[int, dec2])

        self.assertEqual(S[Annotated[int, dec]], Annotated[int, dec, dec2])
        L = Annotated[List[T], dec]

        self.assertEqual(L[int], Annotated[List[int], dec])
        with self.assertRaises(TypeError):
            L[int, int]

        self.assertEqual(S[L[int]], Annotated[List[int], dec, dec2])

        D = Annotated[Dict[KT, VT], dec]
        self.assertEqual(D[str, int], Annotated[Dict[str, int], dec])
        with self.assertRaises(TypeError):
            D[int]

        It = Annotated[int, dec]
        with self.assertRaises(TypeError):
            It[None]

        LI = L[int]
        with self.assertRaises(TypeError):
            LI[None]

    def test_annotated_in_other_types(self):
        X = List[Annotated[T, 5]]
        self.assertEqual(X[int], List[Annotated[int, 5]])


class TypeAliasTests(BaseTestCase):
    def test_canonical_usage(self):
        Alias = Employee  # type: TypeAlias

    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            TypeAlias()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(42, TypeAlias)

    def test_no_issubclass(self):
        with self.assertRaises(TypeError):
            issubclass(Employee, TypeAlias)

        with self.assertRaises(TypeError):
            issubclass(TypeAlias, Employee)

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(TypeAlias):
                pass

        with self.assertRaises(TypeError):
            class C(type(TypeAlias)):
                pass

    def test_repr(self):
        if hasattr(typing, 'TypeAlias'):
            self.assertEqual(repr(TypeAlias), 'typing.TypeAlias')
            self.assertEqual(repr(type(TypeAlias)), 'typing.TypeAlias')
        else:
            self.assertEqual(repr(TypeAlias), 'typing_extensions.TypeAlias')
            self.assertEqual(repr(type(TypeAlias)), 'typing_extensions.TypeAlias')

    def test_cannot_subscript(self):
        with self.assertRaises(TypeError):
            TypeAlias[int]


class AllTests(BaseTestCase):

    def test_typing_extensions_includes_standard(self):
        a = typing_extensions.__all__
        self.assertIn('ClassVar', a)
        self.assertIn('Type', a)
        self.assertIn('Counter', a)
        self.assertIn('DefaultDict', a)
        self.assertIn('Deque', a)
        self.assertIn('NewType', a)
        self.assertIn('overload', a)
        self.assertIn('Text', a)
        self.assertIn('TYPE_CHECKING', a)

    def test_typing_extensions_defers_when_possible(self):
        exclude = {'overload', 'Text', 'TYPE_CHECKING', 'Final'}
        for item in typing_extensions.__all__:
            if item not in exclude and hasattr(typing, item):
                self.assertIs(
                    getattr(typing_extensions, item),
                    getattr(typing, item))

    # def test_typing_extensions_compiles_with_opt(self):
    #     file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                              'typing_extensions.py')
    #     try:
    #         subprocess.check_output('{} -OO {}'.format(sys.executable,
    #                                                    file_path),
    #                                 stderr=subprocess.STDOUT,
    #                                 shell=True)
    #     except subprocess.CalledProcessError:
    #         self.fail('Module does not compile with optimize=2 (-OO flag).')


if __name__ == '__main__':
    main()
