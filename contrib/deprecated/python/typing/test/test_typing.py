from __future__ import absolute_import, unicode_literals

import collections
import contextlib
import os
import pickle
import re
import subprocess
import sys
import abc
import types
from unittest import TestCase, main, SkipTest
from copy import copy, deepcopy

from typing import Any, NoReturn
from typing import TypeVar, AnyStr
from typing import T, KT, VT  # Not in __all__.
from typing import Union, Optional
from typing import Tuple, List, MutableMapping
from typing import Callable
from typing import Generic, ClassVar, GenericMeta, Final, Literal
from typing import cast
from typing import Type, Protocol, runtime_checkable
from typing import NewType
from typing import NamedTuple, TypedDict
from typing import Pattern, Match
import typing
import weakref
import collections


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

    def clear_caches(self):
        for f in typing._cleanups:
            f()


class Employee(object):
    pass


class Manager(Employee):
    pass


class Founder(Employee):
    pass


class ManagingFounder(Manager, Founder):
    pass


class AnyTests(BaseTestCase):

    def test_any_instance_type_error(self):
        with self.assertRaises(TypeError):
            isinstance(42, Any)

    def test_any_subclass_type_error(self):
        with self.assertRaises(TypeError):
            issubclass(Employee, Any)
        with self.assertRaises(TypeError):
            issubclass(Any, Employee)

    def test_repr(self):
        self.assertEqual(repr(Any), 'typing.Any')

    def test_errors(self):
        with self.assertRaises(TypeError):
            issubclass(42, Any)
        with self.assertRaises(TypeError):
            Any[int]  # Any is not a generic type.

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class A(Any):
                pass
        with self.assertRaises(TypeError):
            class A(type(Any)):
                pass

    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            Any()
        with self.assertRaises(TypeError):
            type(Any)()

    def test_any_is_subclass(self):
        # These expressions must simply not fail.
        typing.Match[Any]
        typing.Pattern[Any]
        typing.IO[Any]


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
        self.assertEqual(repr(NoReturn), 'typing.NoReturn')

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


class TypeVarTests(BaseTestCase):

    def test_basic_plain(self):
        T = TypeVar('T')
        # T equals itself.
        self.assertEqual(T, T)
        # T is an instance of TypeVar
        self.assertIsInstance(T, TypeVar)

    def test_typevar_instance_type_error(self):
        T = TypeVar('T')
        with self.assertRaises(TypeError):
            isinstance(42, T)

    def test_typevar_subclass_type_error(self):
        T = TypeVar('T')
        with self.assertRaises(TypeError):
            issubclass(int, T)
        with self.assertRaises(TypeError):
            issubclass(T, int)

    def test_constrained_error(self):
        with self.assertRaises(TypeError):
            X = TypeVar('X', int)
            X

    def test_union_unique(self):
        X = TypeVar('X')
        Y = TypeVar('Y')
        self.assertNotEqual(X, Y)
        self.assertEqual(Union[X], X)
        self.assertNotEqual(Union[X], Union[X, Y])
        self.assertEqual(Union[X, X], X)
        self.assertNotEqual(Union[X, int], Union[X])
        self.assertNotEqual(Union[X, int], Union[int])
        self.assertEqual(Union[X, int].__args__, (X, int))
        self.assertEqual(Union[X, int].__parameters__, (X,))
        self.assertIs(Union[X, int].__origin__, Union)

    def test_union_constrained(self):
        A = TypeVar('A', str, bytes)
        self.assertNotEqual(Union[A, str], Union[A])

    def test_repr(self):
        self.assertEqual(repr(T), '~T')
        self.assertEqual(repr(KT), '~KT')
        self.assertEqual(repr(VT), '~VT')
        self.assertEqual(repr(AnyStr), '~AnyStr')
        T_co = TypeVar('T_co', covariant=True)
        self.assertEqual(repr(T_co), '+T_co')
        T_contra = TypeVar('T_contra', contravariant=True)
        self.assertEqual(repr(T_contra), '-T_contra')

    def test_no_redefinition(self):
        self.assertNotEqual(TypeVar('T'), TypeVar('T'))
        self.assertNotEqual(TypeVar('T', int, str), TypeVar('T', int, str))

    def test_cannot_subclass_vars(self):
        with self.assertRaises(TypeError):
            class V(TypeVar('T')):
                pass

    def test_cannot_subclass_var_itself(self):
        with self.assertRaises(TypeError):
            class V(TypeVar):
                pass

    def test_cannot_instantiate_vars(self):
        with self.assertRaises(TypeError):
            TypeVar('A')()

    def test_bound_errors(self):
        with self.assertRaises(TypeError):
            TypeVar('X', bound=42)
        with self.assertRaises(TypeError):
            TypeVar('X', str, float, bound=Employee)

    def test_no_bivariant(self):
        with self.assertRaises(ValueError):
            TypeVar('T', covariant=True, contravariant=True)


class UnionTests(BaseTestCase):

    def test_basics(self):
        u = Union[int, float]
        self.assertNotEqual(u, Union)

    def test_subclass_error(self):
        with self.assertRaises(TypeError):
            issubclass(int, Union)
        with self.assertRaises(TypeError):
            issubclass(Union, int)
        with self.assertRaises(TypeError):
            issubclass(int, Union[int, str])
        with self.assertRaises(TypeError):
            issubclass(Union[int, str], int)

    def test_union_any(self):
        u = Union[Any]
        self.assertEqual(u, Any)
        u1 = Union[int, Any]
        u2 = Union[Any, int]
        u3 = Union[Any, object]
        self.assertEqual(u1, u2)
        self.assertNotEqual(u1, Any)
        self.assertNotEqual(u2, Any)
        self.assertNotEqual(u3, Any)

    def test_union_object(self):
        u = Union[object]
        self.assertEqual(u, object)
        u = Union[int, object]
        self.assertEqual(u, object)
        u = Union[object, int]
        self.assertEqual(u, object)

    def test_unordered(self):
        u1 = Union[int, float]
        u2 = Union[float, int]
        self.assertEqual(u1, u2)

    def test_single_class_disappears(self):
        t = Union[Employee]
        self.assertIs(t, Employee)

    def test_base_class_disappears(self):
        u = Union[Employee, Manager, int]
        self.assertEqual(u, Union[int, Employee])
        u = Union[Manager, int, Employee]
        self.assertEqual(u, Union[int, Employee])
        u = Union[Employee, Manager]
        self.assertIs(u, Employee)

    def test_union_union(self):
        u = Union[int, float]
        v = Union[u, Employee]
        self.assertEqual(v, Union[int, float, Employee])

    def test_repr(self):
        self.assertEqual(repr(Union), 'typing.Union')
        u = Union[Employee, int]
        self.assertEqual(repr(u), 'typing.Union[%s.Employee, int]' % __name__)
        u = Union[int, Employee]
        self.assertEqual(repr(u), 'typing.Union[int, %s.Employee]' % __name__)
        T = TypeVar('T')
        u = Union[T, int][int]
        self.assertEqual(repr(u), repr(int))
        u = Union[List[int], int]
        self.assertEqual(repr(u), 'typing.Union[typing.List[int], int]')

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(Union):
                pass
        with self.assertRaises(TypeError):
            class C(type(Union)):
                pass
        with self.assertRaises(TypeError):
            class C(Union[int, str]):
                pass

    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            Union()
        u = Union[int, float]
        with self.assertRaises(TypeError):
            u()
        with self.assertRaises(TypeError):
            type(u)()

    def test_union_generalization(self):
        self.assertFalse(Union[str, typing.Iterable[int]] == str)
        self.assertFalse(Union[str, typing.Iterable[int]] == typing.Iterable[int])
        self.assertTrue(Union[str, typing.Iterable] == typing.Iterable)

    def test_union_compare_other(self):
        self.assertNotEqual(Union, object)
        self.assertNotEqual(Union, Any)
        self.assertNotEqual(ClassVar, Union)
        self.assertNotEqual(Optional, Union)
        self.assertNotEqual([None], Optional)
        self.assertNotEqual(Optional, typing.Mapping)
        self.assertNotEqual(Optional[typing.MutableMapping], Union)

    def test_optional(self):
        o = Optional[int]
        u = Union[int, None]
        self.assertEqual(o, u)

    def test_empty(self):
        with self.assertRaises(TypeError):
            Union[()]

    def test_union_instance_type_error(self):
        with self.assertRaises(TypeError):
            isinstance(42, Union[int, str])

    def test_no_eval_union(self):
        u = Union[int, str]
        self.assertIs(u._eval_type({}, {}), u)

    def test_function_repr_union(self):
        def fun(): pass
        self.assertEqual(repr(Union[fun, int]), 'typing.Union[fun, int]')

    def test_union_str_pattern(self):
        # Shouldn't crash; see http://bugs.python.org/issue25390
        A = Union[str, Pattern]
        A

    def test_etree(self):
        # See https://github.com/python/typing/issues/229
        # (Only relevant for Python 2.)
        try:
            from xml.etree.cElementTree import Element
        except ImportError:
            raise SkipTest("cElementTree not found")
        Union[Element, str]  # Shouldn't crash

        def Elem(*args):
            return Element(*args)

        Union[Elem, str]  # Nor should this


class TupleTests(BaseTestCase):

    def test_basics(self):
        with self.assertRaises(TypeError):
            issubclass(Tuple, Tuple[int, str])
        with self.assertRaises(TypeError):
            issubclass(tuple, Tuple[int, str])

        class TP(tuple): pass
        self.assertTrue(issubclass(tuple, Tuple))
        self.assertTrue(issubclass(TP, Tuple))

    def test_equality(self):
        self.assertEqual(Tuple[int], Tuple[int])
        self.assertEqual(Tuple[int, ...], Tuple[int, ...])
        self.assertNotEqual(Tuple[int], Tuple[int, int])
        self.assertNotEqual(Tuple[int], Tuple[int, ...])

    def test_tuple_subclass(self):
        class MyTuple(tuple):
            pass
        self.assertTrue(issubclass(MyTuple, Tuple))

    def test_tuple_instance_type_error(self):
        with self.assertRaises(TypeError):
            isinstance((0, 0), Tuple[int, int])
        isinstance((0, 0), Tuple)

    def test_repr(self):
        self.assertEqual(repr(Tuple), 'typing.Tuple')
        self.assertEqual(repr(Tuple[()]), 'typing.Tuple[()]')
        self.assertEqual(repr(Tuple[int, float]), 'typing.Tuple[int, float]')
        self.assertEqual(repr(Tuple[int, ...]), 'typing.Tuple[int, ...]')

    def test_errors(self):
        with self.assertRaises(TypeError):
            issubclass(42, Tuple)
        with self.assertRaises(TypeError):
            issubclass(42, Tuple[int])


class CallableTests(BaseTestCase):

    def test_self_subclass(self):
        with self.assertRaises(TypeError):
            self.assertTrue(issubclass(type(lambda x: x), Callable[[int], int]))
        self.assertTrue(issubclass(type(lambda x: x), Callable))

    def test_eq_hash(self):
        self.assertEqual(Callable[[int], int], Callable[[int], int])
        self.assertEqual(len({Callable[[int], int], Callable[[int], int]}), 1)
        self.assertNotEqual(Callable[[int], int], Callable[[int], str])
        self.assertNotEqual(Callable[[int], int], Callable[[str], int])
        self.assertNotEqual(Callable[[int], int], Callable[[int, int], int])
        self.assertNotEqual(Callable[[int], int], Callable[[], int])
        self.assertNotEqual(Callable[[int], int], Callable)

    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            Callable()
        with self.assertRaises(TypeError):
            type(Callable)()
        c = Callable[[int], str]
        with self.assertRaises(TypeError):
            c()
        with self.assertRaises(TypeError):
            type(c)()

    def test_callable_wrong_forms(self):
        with self.assertRaises(TypeError):
            Callable[(), int]
        with self.assertRaises(TypeError):
            Callable[[()], int]
        with self.assertRaises(TypeError):
            Callable[[int, 1], 2]
        with self.assertRaises(TypeError):
            Callable[int]

    def test_callable_instance_works(self):
        def f():
            pass
        self.assertIsInstance(f, Callable)
        self.assertNotIsInstance(None, Callable)

    def test_callable_instance_type_error(self):
        def f():
            pass
        with self.assertRaises(TypeError):
            self.assertIsInstance(f, Callable[[], None])
        with self.assertRaises(TypeError):
            self.assertIsInstance(f, Callable[[], Any])
        with self.assertRaises(TypeError):
            self.assertNotIsInstance(None, Callable[[], None])
        with self.assertRaises(TypeError):
            self.assertNotIsInstance(None, Callable[[], Any])

    def test_repr(self):
        ct0 = Callable[[], bool]
        self.assertEqual(repr(ct0), 'typing.Callable[[], bool]')
        ct2 = Callable[[str, float], int]
        self.assertEqual(repr(ct2), 'typing.Callable[[str, float], int]')
        ctv = Callable[..., str]
        self.assertEqual(repr(ctv), 'typing.Callable[..., str]')

    def test_ellipsis_in_generic(self):
        # Shouldn't crash; see https://github.com/python/typing/issues/259
        typing.List[Callable[..., str]]


XK = TypeVar('XK', unicode, bytes)
XV = TypeVar('XV')


class SimpleMapping(Generic[XK, XV]):

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    def get(self, key, default=None):
        pass


class MySimpleMapping(SimpleMapping[XK, XV]):

    def __init__(self):
        self.store = {}

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def get(self, key, default=None):
        try:
            return self.store[key]
        except KeyError:
            return default


class ProtocolTests(BaseTestCase):

    def test_basic_protocol(self):
        @runtime_checkable
        class P(Protocol):
            def meth(self):
                pass
        class C(object): pass
        class D(object):
            def meth(self):
                pass
        def f():
            pass
        self.assertIsSubclass(D, P)
        self.assertIsInstance(D(), P)
        self.assertNotIsSubclass(C, P)
        self.assertNotIsInstance(C(), P)
        self.assertNotIsSubclass(types.FunctionType, P)
        self.assertNotIsInstance(f, P)

    def test_everything_implements_empty_protocol(self):
        @runtime_checkable
        class Empty(Protocol): pass
        class C(object): pass
        def f():
            pass
        for thing in (object, type, tuple, C, types.FunctionType):
            self.assertIsSubclass(thing, Empty)
        for thing in (object(), 1, (), typing, f):
            self.assertIsInstance(thing, Empty)

    def test_function_implements_protocol(self):
        @runtime_checkable
        class Function(Protocol):
            def __call__(self, *args, **kwargs):
                pass
        def f():
            pass
        self.assertIsInstance(f, Function)

    def test_no_inheritance_from_nominal(self):
        class C(object): pass
        class BP(Protocol): pass
        with self.assertRaises(TypeError):
            class P(C, Protocol):
                pass
        with self.assertRaises(TypeError):
            class P(Protocol, C):
                pass
        with self.assertRaises(TypeError):
            class P(BP, C, Protocol):
                pass
        class D(BP, C): pass
        class E(C, BP): pass
        self.assertNotIsInstance(D(), E)
        self.assertNotIsInstance(E(), D)

    def test_no_instantiation(self):
        class P(Protocol): pass
        with self.assertRaises(TypeError):
            P()
        class C(P): pass
        self.assertIsInstance(C(), C)
        T = typing.TypeVar('T')
        class PG(Protocol[T]): pass
        with self.assertRaises(TypeError):
            PG()
        with self.assertRaises(TypeError):
            PG[int]()
        with self.assertRaises(TypeError):
            PG[T]()
        class CG(PG[T]): pass
        self.assertIsInstance(CG[int](), CG)

    def test_cannot_instantiate_abstract(self):
        @runtime_checkable
        class P(Protocol):
            @abc.abstractmethod
            def ameth(self):
                raise NotImplementedError
        class B(P):
            pass
        class C(B):
            def ameth(self):
                return 26
        with self.assertRaises(TypeError):
            B()
        self.assertIsInstance(C(), P)

    def test_subprotocols_extending(self):
        class P1(Protocol):
            def meth1(self):
                pass
        @runtime_checkable
        class P2(P1, Protocol):
            def meth2(self):
                pass
        class C(object):
            def meth1(self):
                pass
            def meth2(self):
                pass
        class C1(object):
            def meth1(self):
                pass
        class C2(object):
            def meth2(self):
                pass
        self.assertNotIsInstance(C1(), P2)
        self.assertNotIsInstance(C2(), P2)
        self.assertNotIsSubclass(C1, P2)
        self.assertNotIsSubclass(C2, P2)
        self.assertIsInstance(C(), P2)
        self.assertIsSubclass(C, P2)

    def test_subprotocols_merging(self):
        class P1(Protocol):
            def meth1(self):
                pass
        class P2(Protocol):
            def meth2(self):
                pass
        @runtime_checkable
        class P(P1, P2, Protocol):
            pass
        class C(object):
            def meth1(self):
                pass
            def meth2(self):
                pass
        class C1(object):
            def meth1(self):
                pass
        class C2(object):
            def meth2(self):
                pass
        self.assertNotIsInstance(C1(), P)
        self.assertNotIsInstance(C2(), P)
        self.assertNotIsSubclass(C1, P)
        self.assertNotIsSubclass(C2, P)
        self.assertIsInstance(C(), P)
        self.assertIsSubclass(C, P)

    def test_protocols_issubclass(self):
        T = typing.TypeVar('T')
        @runtime_checkable
        class P(Protocol):
            def x(self): pass
        @runtime_checkable
        class PG(Protocol[T]):
            def x(self): pass
        class BadP(Protocol):
            def x(self): pass
        class BadPG(Protocol[T]):
            def x(self): pass
        class C(object):
            def x(self): pass
        self.assertIsSubclass(C, P)
        self.assertIsSubclass(C, PG)
        self.assertIsSubclass(BadP, PG)
        self.assertIsSubclass(PG[int], PG)
        self.assertIsSubclass(BadPG[int], P)
        self.assertIsSubclass(BadPG[T], PG)
        with self.assertRaises(TypeError):
            issubclass(C, PG[T])
        with self.assertRaises(TypeError):
            issubclass(C, PG[C])
        with self.assertRaises(TypeError):
            issubclass(C, BadP)
        with self.assertRaises(TypeError):
            issubclass(C, BadPG)
        with self.assertRaises(TypeError):
            issubclass(P, PG[T])
        with self.assertRaises(TypeError):
            issubclass(PG, PG[int])

    def test_protocols_issubclass_non_callable(self):
        class C(object):
            x = 1
        @runtime_checkable
        class PNonCall(Protocol):
            x = 1
        with self.assertRaises(TypeError):
            issubclass(C, PNonCall)
        self.assertIsInstance(C(), PNonCall)
        PNonCall.register(C)
        with self.assertRaises(TypeError):
            issubclass(C, PNonCall)
        self.assertIsInstance(C(), PNonCall)
        # check that non-protocol subclasses are not affected
        class D(PNonCall): pass
        self.assertNotIsSubclass(C, D)
        self.assertNotIsInstance(C(), D)
        D.register(C)
        self.assertIsSubclass(C, D)
        self.assertIsInstance(C(), D)
        with self.assertRaises(TypeError):
            issubclass(D, PNonCall)

    def test_protocols_isinstance(self):
        T = typing.TypeVar('T')
        @runtime_checkable
        class P(Protocol):
            def meth(x): pass
        @runtime_checkable
        class PG(Protocol[T]):
            def meth(x): pass
        class BadP(Protocol):
            def meth(x): pass
        class BadPG(Protocol[T]):
            def meth(x): pass
        class C(object):
            def meth(x): pass
        self.assertIsInstance(C(), P)
        self.assertIsInstance(C(), PG)
        with self.assertRaises(TypeError):
            isinstance(C(), PG[T])
        with self.assertRaises(TypeError):
            isinstance(C(), PG[C])
        with self.assertRaises(TypeError):
            isinstance(C(), BadP)
        with self.assertRaises(TypeError):
            isinstance(C(), BadPG)

    def test_protocols_isinstance_init(self):
        T = typing.TypeVar('T')
        @runtime_checkable
        class P(Protocol):
            x = 1
        @runtime_checkable
        class PG(Protocol[T]):
            x = 1
        class C(object):
            def __init__(self, x):
                self.x = x
        self.assertIsInstance(C(1), P)
        self.assertIsInstance(C(1), PG)

    def test_protocol_checks_after_subscript(self):
        class P(Protocol[T]): pass
        class C(P[T]): pass
        class Old1: pass
        class New1(object): pass
        class Old2: pass
        class New2(object): pass
        CA = C[Any]  # noqa

        self.assertNotIsInstance(Old1(), C)
        self.assertNotIsInstance(New1(), C)
        self.assertNotIsSubclass(Old2, C)
        self.assertNotIsSubclass(New2, C)

        class D1(C[Any]): pass
        class D2(C[Any]): pass
        CI = C[int]  # noqa

        self.assertIsInstance(D1(), C)
        self.assertIsSubclass(D2, C)

    def test_protocols_support_register(self):
        @runtime_checkable
        class P(Protocol):
            x = 1
        class PM(Protocol):
            def meth(self): pass
        class D(PM): pass
        class C(object): pass
        D.register(C)
        P.register(C)
        self.assertIsInstance(C(), P)
        self.assertIsInstance(C(), D)

    def test_none_on_non_callable_doesnt_block_implementation(self):
        @runtime_checkable
        class P(Protocol):
            x = 1
        class A(object):
            x = 1
        class B(A):
            x = None
        class C(object):
            def __init__(self):
                self.x = None
        self.assertIsInstance(B(), P)
        self.assertIsInstance(C(), P)

    def test_none_on_callable_blocks_implementation(self):
        @runtime_checkable
        class P(Protocol):
            def x(self): pass
        class A(object):
            def x(self): pass
        class B(A):
            x = None
        class C(object):
            def __init__(self):
                self.x = None
        self.assertNotIsInstance(B(), P)
        self.assertNotIsInstance(C(), P)

    def test_non_protocol_subclasses(self):
        class P(Protocol):
            x = 1
        @runtime_checkable
        class PR(Protocol):
            def meth(self): pass
        class NonP(P):
            x = 1
        class NonPR(PR): pass
        class C(object):
            x = 1
        class D(object):
            def meth(self): pass
        self.assertNotIsInstance(C(), NonP)
        self.assertNotIsInstance(D(), NonPR)
        self.assertNotIsSubclass(C, NonP)
        self.assertNotIsSubclass(D, NonPR)
        self.assertIsInstance(NonPR(), PR)
        self.assertIsSubclass(NonPR, PR)

    def test_custom_subclasshook(self):
        class P(Protocol):
            x = 1
        class OKClass(object): pass
        class BadClass(object):
            x = 1
        class C(P):
            @classmethod
            def __subclasshook__(cls, other):
                return other.__name__.startswith("OK")
        self.assertIsInstance(OKClass(), C)
        self.assertNotIsInstance(BadClass(), C)
        self.assertIsSubclass(OKClass, C)
        self.assertNotIsSubclass(BadClass, C)

    def test_issubclass_fails_correctly(self):
        @runtime_checkable
        class P(Protocol):
            x = 1
        class C: pass
        with self.assertRaises(TypeError):
            issubclass(C(), P)

    def test_defining_generic_protocols(self):
        T = typing.TypeVar('T')
        S = typing.TypeVar('S')
        @runtime_checkable
        class PR(Protocol[T, S]):
            def meth(self): pass
        class P(PR[int, T], Protocol[T]):
            y = 1
        self.assertIsSubclass(PR[int, T], PR)
        self.assertIsSubclass(P[str], PR)
        with self.assertRaises(TypeError):
            PR[int]
        with self.assertRaises(TypeError):
            P[int, str]
        with self.assertRaises(TypeError):
            PR[int, 1]
        with self.assertRaises(TypeError):
            PR[int, ClassVar]
        class C(PR[int, T]): pass
        self.assertIsInstance(C[str](), C)

    def test_defining_generic_protocols_old_style(self):
        T = typing.TypeVar('T')
        S = typing.TypeVar('S')
        @runtime_checkable
        class PR(Protocol, typing.Generic[T, S]):
            def meth(self): pass
        class P(PR[int, str], Protocol):
            y = 1
        self.assertIsSubclass(PR[int, str], PR)
        self.assertIsSubclass(P, PR)
        with self.assertRaises(TypeError):
            PR[int]
        with self.assertRaises(TypeError):
            PR[int, 1]
        class P1(Protocol, typing.Generic[T]):
            def bar(self, x): pass
        class P2(typing.Generic[T], Protocol):
            def bar(self, x): pass
        @runtime_checkable
        class PSub(P1[str], Protocol):
            x = 1
        class Test(object):
            x = 1
            def bar(self, x):
                return x
        self.assertIsInstance(Test(), PSub)
        with self.assertRaises(TypeError):
            PR[int, ClassVar]

    def test_init_called(self):
        T = typing.TypeVar('T')
        class P(Protocol[T]): pass
        class C(P[T]):
            def __init__(self):
                self.test = 'OK'
        self.assertEqual(C[int]().test, 'OK')

    def test_protocols_bad_subscripts(self):
        T = typing.TypeVar('T')
        S = typing.TypeVar('S')
        with self.assertRaises(TypeError):
            class P(Protocol[T, T]): pass
        with self.assertRaises(TypeError):
            class P(Protocol[int]): pass
        with self.assertRaises(TypeError):
            class P(Protocol[T], Protocol[S]): pass
        with self.assertRaises(TypeError):
            class P(Protocol[T], typing.Mapping[T, S]): pass

    def test_generic_protocols_repr(self):
        T = typing.TypeVar('T')
        S = typing.TypeVar('S')
        class P(Protocol[T, S]): pass
        self.assertTrue(repr(P).endswith('P'))
        self.assertTrue(repr(P[T, S]).endswith('P[~T, ~S]'))
        self.assertTrue(repr(P[int, str]).endswith('P[int, str]'))

    def test_generic_protocols_eq(self):
        T = typing.TypeVar('T')
        S = typing.TypeVar('S')
        class P(Protocol[T, S]): pass
        self.assertEqual(P, P)
        self.assertEqual(P[int, T], P[int, T])
        self.assertEqual(P[T, T][typing.Tuple[T, S]][int, str],
                         P[typing.Tuple[int, str], typing.Tuple[int, str]])

    def test_generic_protocols_special_from_generic(self):
        T = typing.TypeVar('T')
        class P(Protocol[T]): pass
        self.assertEqual(P.__parameters__, (T,))
        self.assertIs(P.__args__, None)
        self.assertIs(P.__origin__, None)
        self.assertEqual(P[int].__parameters__, ())
        self.assertEqual(P[int].__args__, (int,))
        self.assertIs(P[int].__origin__, P)

    def test_generic_protocols_special_from_protocol(self):
        @runtime_checkable
        class PR(Protocol):
            x = 1
        class P(Protocol):
            def meth(self):
                pass
        T = typing.TypeVar('T')
        class PG(Protocol[T]):
            x = 1
            def meth(self):
                pass
        self.assertTrue(P._is_protocol)
        self.assertTrue(PR._is_protocol)
        self.assertTrue(PG._is_protocol)
        with self.assertRaises(AttributeError):
            self.assertFalse(P._is_runtime_protocol)
        self.assertTrue(PR._is_runtime_protocol)
        self.assertTrue(PG[int]._is_protocol)
        self.assertEqual(P._get_protocol_attrs(), {'meth'})
        self.assertEqual(PR._get_protocol_attrs(), {'x'})
        self.assertEqual(frozenset(PG._get_protocol_attrs()),
                         frozenset({'x', 'meth'}))
        self.assertEqual(frozenset(PG[int]._get_protocol_attrs()),
                         frozenset({'x', 'meth'}))

    def test_no_runtime_deco_on_nominal(self):
        with self.assertRaises(TypeError):
            @runtime_checkable
            class C(object): pass
        class Proto(Protocol):
            x = 1
        with self.assertRaises(TypeError):
            @runtime_checkable
            class Concrete(Proto):
                pass

    def test_none_treated_correctly(self):
        @runtime_checkable
        class P(Protocol):
            x = None  # type: int
        class B(object): pass
        self.assertNotIsInstance(B(), P)
        class C(object):
            x = 1
        class D(object):
            x = None
        self.assertIsInstance(C(), P)
        self.assertIsInstance(D(), P)
        class CI(object):
            def __init__(self):
                self.x = 1
        class DI(object):
            def __init__(self):
                self.x = None
        self.assertIsInstance(C(), P)
        self.assertIsInstance(D(), P)

    def test_protocols_in_unions(self):
        class P(Protocol):
            x = None  # type: int
        Alias = typing.Union[typing.Iterable, P]
        Alias2 = typing.Union[P, typing.Iterable]
        self.assertEqual(Alias, Alias2)

    def test_protocols_pickleable(self):
        global P, CP  # pickle wants to reference the class by name
        T = typing.TypeVar('T')

        @runtime_checkable
        class P(Protocol[T]):
            x = 1
        class CP(P[int]):
            pass

        c = CP()
        c.foo = 42
        c.bar = 'abc'
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z = pickle.dumps(c, proto)
            x = pickle.loads(z)
            self.assertEqual(x.foo, 42)
            self.assertEqual(x.bar, 'abc')
            self.assertEqual(x.x, 1)
            self.assertEqual(x.__dict__, {'foo': 42, 'bar': 'abc'})
            s = pickle.dumps(P)
            D = pickle.loads(s)
            class E(object):
                x = 1
            self.assertIsInstance(E(), D)

    def test_supports_int(self):
        self.assertIsSubclass(int, typing.SupportsInt)
        self.assertNotIsSubclass(str, typing.SupportsInt)

    def test_supports_float(self):
        self.assertIsSubclass(float, typing.SupportsFloat)
        self.assertNotIsSubclass(str, typing.SupportsFloat)

    def test_supports_complex(self):

        # Note: complex itself doesn't have __complex__.
        class C(object):
            def __complex__(self):
                return 0j

        self.assertIsSubclass(C, typing.SupportsComplex)
        self.assertNotIsSubclass(str, typing.SupportsComplex)

    def test_supports_abs(self):
        self.assertIsSubclass(float, typing.SupportsAbs)
        self.assertIsSubclass(int, typing.SupportsAbs)
        self.assertNotIsSubclass(str, typing.SupportsAbs)

    def test_reversible(self):
        self.assertIsSubclass(list, typing.Reversible)
        self.assertNotIsSubclass(int, typing.Reversible)

    def test_supports_index(self):
        self.assertIsSubclass(int, typing.SupportsIndex)
        self.assertNotIsSubclass(str, typing.SupportsIndex)

    def test_protocol_instance_works(self):
        self.assertIsInstance(0, typing.SupportsAbs)
        self.assertNotIsInstance('no', typing.SupportsAbs)
        class C1(typing.SupportsInt):
            def __int__(self):
                return 42
        class C2(C1):
            pass
        c = C2()
        self.assertIsInstance(c, C1)

    def test_collections_protocols_allowed(self):
        @runtime_checkable
        class Custom(collections.Iterable, Protocol):
            def close(self): pass

        class A(object): pass
        class B(object):
            def __iter__(self):
                return []
            def close(self):
                return 0

        self.assertIsSubclass(B, Custom)
        self.assertNotIsSubclass(A, Custom)


class GenericTests(BaseTestCase):

    def test_basics(self):
        X = SimpleMapping[str, Any]
        self.assertEqual(X.__parameters__, ())
        with self.assertRaises(TypeError):
            X[unicode]
        with self.assertRaises(TypeError):
            X[unicode, unicode]
        Y = SimpleMapping[XK, unicode]
        self.assertEqual(Y.__parameters__, (XK,))
        Y[unicode]
        with self.assertRaises(TypeError):
            Y[unicode, unicode]
        self.assertIsSubclass(SimpleMapping[str, int], SimpleMapping)

    def test_generic_errors(self):
        T = TypeVar('T')
        S = TypeVar('S')
        with self.assertRaises(TypeError):
            Generic[T]()
        with self.assertRaises(TypeError):
            Generic[T][T]
        with self.assertRaises(TypeError):
            Generic[T][S]
        with self.assertRaises(TypeError):
            isinstance([], List[int])
        with self.assertRaises(TypeError):
            issubclass(list, List[int])
        with self.assertRaises(TypeError):
            class NewGeneric(Generic): pass
        with self.assertRaises(TypeError):
            class MyGeneric(Generic[T], Generic[S]): pass
        with self.assertRaises(TypeError):
            class MyGeneric(List[T], Generic[S]): pass

    def test_init(self):
        T = TypeVar('T')
        S = TypeVar('S')
        with self.assertRaises(TypeError):
            Generic[T, T]
        with self.assertRaises(TypeError):
            Generic[T, S, T]

    def test_repr(self):
        self.assertEqual(repr(SimpleMapping),
                         __name__ + '.' + 'SimpleMapping')
        self.assertEqual(repr(MySimpleMapping),
                         __name__ + '.' + 'MySimpleMapping')

    def test_chain_repr(self):
        T = TypeVar('T')
        S = TypeVar('S')

        class C(Generic[T]):
            pass

        X = C[Tuple[S, T]]
        self.assertEqual(X, C[Tuple[S, T]])
        self.assertNotEqual(X, C[Tuple[T, S]])

        Y = X[T, int]
        self.assertEqual(Y, X[T, int])
        self.assertNotEqual(Y, X[S, int])
        self.assertNotEqual(Y, X[T, str])

        Z = Y[str]
        self.assertEqual(Z, Y[str])
        self.assertNotEqual(Z, Y[int])
        self.assertNotEqual(Z, Y[T])

        self.assertTrue(str(Z).endswith(
            '.C[typing.Tuple[str, int]]'))

    def test_new_repr(self):
        T = TypeVar('T')
        U = TypeVar('U', covariant=True)
        S = TypeVar('S')

        self.assertEqual(repr(List), 'typing.List')
        self.assertEqual(repr(List[T]), 'typing.List[~T]')
        self.assertEqual(repr(List[U]), 'typing.List[+U]')
        self.assertEqual(repr(List[S][T][int]), 'typing.List[int]')
        self.assertEqual(repr(List[int]), 'typing.List[int]')

    def test_new_repr_complex(self):
        T = TypeVar('T')
        TS = TypeVar('TS')

        self.assertEqual(repr(typing.Mapping[T, TS][TS, T]), 'typing.Mapping[~TS, ~T]')
        self.assertEqual(repr(List[Tuple[T, TS]][int, T]),
                         'typing.List[typing.Tuple[int, ~T]]')
        self.assertEqual(
            repr(List[Tuple[T, T]][List[int]]),
            'typing.List[typing.Tuple[typing.List[int], typing.List[int]]]'
        )

    def test_new_repr_bare(self):
        T = TypeVar('T')
        self.assertEqual(repr(Generic[T]), 'typing.Generic[~T]')
        self.assertEqual(repr(typing.Protocol[T]), 'typing.Protocol[~T]')
        class C(typing.Dict[Any, Any]): pass
        # this line should just work
        repr(C.__mro__)

    def test_dict(self):
        T = TypeVar('T')

        class B(Generic[T]):
            pass

        b = B()
        b.foo = 42
        self.assertEqual(b.__dict__, {'foo': 42})

        class C(B[int]):
            pass

        c = C()
        c.bar = 'abc'
        self.assertEqual(c.__dict__, {'bar': 'abc'})

    def test_subscripted_generics_as_proxies(self):
        T = TypeVar('T')
        class C(Generic[T]):
            x = 'def'
        self.assertEqual(C[int].x, 'def')
        self.assertEqual(C[C[int]].x, 'def')
        C[C[int]].x = 'changed'
        self.assertEqual(C.x, 'changed')
        self.assertEqual(C[str].x, 'changed')
        C[List[str]].z = 'new'
        self.assertEqual(C.z, 'new')
        self.assertEqual(C[Tuple[int]].z, 'new')

        self.assertEqual(C().x, 'changed')
        self.assertEqual(C[Tuple[str]]().z, 'new')

        class D(C[T]):
            pass
        self.assertEqual(D[int].x, 'changed')
        self.assertEqual(D.z, 'new')
        D.z = 'from derived z'
        D[int].x = 'from derived x'
        self.assertEqual(C.x, 'changed')
        self.assertEqual(C[int].z, 'new')
        self.assertEqual(D.x, 'from derived x')
        self.assertEqual(D[str].z, 'from derived z')

    def test_abc_registry_kept(self):
        T = TypeVar('T')
        class C(Generic[T]): pass
        C.register(int)
        self.assertIsInstance(1, C)
        C[int]
        self.assertIsInstance(1, C)

    def test_false_subclasses(self):
        class MyMapping(MutableMapping[str, str]): pass
        self.assertNotIsInstance({}, MyMapping)
        self.assertNotIsSubclass(dict, MyMapping)

    def test_abc_bases(self):
        class MM(MutableMapping[str, str]):
            def __getitem__(self, k):
                return None
            def __setitem__(self, k, v):
                pass
            def __delitem__(self, k):
                pass
            def __iter__(self):
                return iter(())
            def __len__(self):
                return 0
        # this should just work
        MM().update()
        self.assertIsInstance(MM(), collections.MutableMapping)
        self.assertIsInstance(MM(), MutableMapping)
        self.assertNotIsInstance(MM(), List)
        self.assertNotIsInstance({}, MM)

    def test_multiple_bases(self):
        class MM1(MutableMapping[str, str], collections.MutableMapping):
            pass
        with self.assertRaises(TypeError):
            # consistent MRO not possible
            class MM2(collections.MutableMapping, MutableMapping[str, str]):
                pass

    def test_orig_bases(self):
        T = TypeVar('T')
        class C(typing.Dict[str, T]): pass
        self.assertEqual(C.__orig_bases__, (typing.Dict[str, T],))

    def test_naive_runtime_checks(self):
        def naive_dict_check(obj, tp):
            # Check if a dictionary conforms to Dict type
            if len(tp.__parameters__) > 0:
                raise NotImplementedError
            if tp.__args__:
                KT, VT = tp.__args__
                return all(
                    isinstance(k, KT) and isinstance(v, VT)
                    for k, v in obj.items()
                )
        self.assertTrue(naive_dict_check({'x': 1}, typing.Dict[typing.Text, int]))
        self.assertFalse(naive_dict_check({1: 'x'}, typing.Dict[typing.Text, int]))
        with self.assertRaises(NotImplementedError):
            naive_dict_check({1: 'x'}, typing.Dict[typing.Text, T])

        def naive_generic_check(obj, tp):
            # Check if an instance conforms to the generic class
            if not hasattr(obj, '__orig_class__'):
                raise NotImplementedError
            return obj.__orig_class__ == tp
        class Node(Generic[T]): pass
        self.assertTrue(naive_generic_check(Node[int](), Node[int]))
        self.assertFalse(naive_generic_check(Node[str](), Node[int]))
        self.assertFalse(naive_generic_check(Node[str](), List))
        with self.assertRaises(NotImplementedError):
            naive_generic_check([1, 2, 3], Node[int])

        def naive_list_base_check(obj, tp):
            # Check if list conforms to a List subclass
            return all(isinstance(x, tp.__orig_bases__[0].__args__[0])
                       for x in obj)
        class C(List[int]): pass
        self.assertTrue(naive_list_base_check([1, 2, 3], C))
        self.assertFalse(naive_list_base_check(['a', 'b'], C))

    def test_multi_subscr_base(self):
        T = TypeVar('T')
        U = TypeVar('U')
        V = TypeVar('V')
        class C(List[T][U][V]): pass
        class D(C, List[T][U][V]): pass
        self.assertEqual(C.__parameters__, (V,))
        self.assertEqual(D.__parameters__, (V,))
        self.assertEqual(C[int].__parameters__, ())
        self.assertEqual(D[int].__parameters__, ())
        self.assertEqual(C[int].__args__, (int,))
        self.assertEqual(D[int].__args__, (int,))
        self.assertEqual(C.__bases__, (List,))
        self.assertEqual(D.__bases__, (C, List))
        self.assertEqual(C.__orig_bases__, (List[T][U][V],))
        self.assertEqual(D.__orig_bases__, (C, List[T][U][V]))

    def test_subscript_meta(self):
        T = TypeVar('T')
        self.assertEqual(Type[GenericMeta], Type[GenericMeta])
        self.assertEqual(Union[T, int][GenericMeta], Union[GenericMeta, int])
        self.assertEqual(Callable[..., GenericMeta].__args__, (Ellipsis, GenericMeta))

    def test_generic_hashes(self):
        import mod_generics_cache
        class A(Generic[T]):
            __module__ = 'test_typing'

        class B(Generic[T]):
            class A(Generic[T]):
                pass

        self.assertEqual(A, A)
        self.assertEqual(mod_generics_cache.A[str], mod_generics_cache.A[str])
        self.assertEqual(B.A, B.A)
        self.assertEqual(mod_generics_cache.B.A[B.A[str]],
                         mod_generics_cache.B.A[B.A[str]])

        self.assertNotEqual(A, B.A)
        self.assertNotEqual(A, mod_generics_cache.A)
        self.assertNotEqual(A, mod_generics_cache.B.A)
        self.assertNotEqual(B.A, mod_generics_cache.A)
        self.assertNotEqual(B.A, mod_generics_cache.B.A)

        self.assertNotEqual(A[str], B.A[str])
        self.assertNotEqual(A[List[Any]], B.A[List[Any]])
        self.assertNotEqual(A[str], mod_generics_cache.A[str])
        self.assertNotEqual(A[str], mod_generics_cache.B.A[str])
        self.assertNotEqual(B.A[int], mod_generics_cache.A[int])
        self.assertNotEqual(B.A[List[Any]], mod_generics_cache.B.A[List[Any]])

        self.assertNotEqual(Tuple[A[str]], Tuple[B.A[str]])
        self.assertNotEqual(Tuple[A[List[Any]]], Tuple[B.A[List[Any]]])
        self.assertNotEqual(Union[str, A[str]], Union[str, mod_generics_cache.A[str]])
        self.assertNotEqual(Union[A[str], A[str]],
                            Union[A[str], mod_generics_cache.A[str]])
        self.assertNotEqual(typing.FrozenSet[A[str]],
                            typing.FrozenSet[mod_generics_cache.B.A[str]])

        self.assertTrue(repr(Tuple[A[str]]).endswith('test_typing.A[str]]'))
        self.assertTrue(repr(Tuple[mod_generics_cache.A[str]])
                        .endswith('mod_generics_cache.A[str]]'))

    def test_extended_generic_rules_eq(self):
        T = TypeVar('T')
        U = TypeVar('U')
        self.assertEqual(Tuple[T, T][int], Tuple[int, int])
        self.assertEqual(typing.Iterable[Tuple[T, T]][T], typing.Iterable[Tuple[T, T]])
        with self.assertRaises(TypeError):
            Tuple[T, int][()]
        with self.assertRaises(TypeError):
            Tuple[T, U][T, ...]

        self.assertEqual(Union[T, int][int], int)
        self.assertEqual(Union[T, U][int, Union[int, str]], Union[int, str])
        class Base(object): pass
        class Derived(Base): pass
        self.assertEqual(Union[T, Base][Derived], Base)
        with self.assertRaises(TypeError):
            Union[T, int][1]

        self.assertEqual(Callable[[T], T][KT], Callable[[KT], KT])
        self.assertEqual(Callable[..., List[T]][int], Callable[..., List[int]])
        with self.assertRaises(TypeError):
            Callable[[T], U][..., int]
        with self.assertRaises(TypeError):
            Callable[[T], U][[], int]

    def test_extended_generic_rules_repr(self):
        T = TypeVar('T')
        self.assertEqual(repr(Union[Tuple, Callable]).replace('typing.', ''),
                         'Union[Tuple, Callable]')
        self.assertEqual(repr(Union[Tuple, Tuple[int]]).replace('typing.', ''),
                         'Tuple')
        self.assertEqual(repr(Callable[..., Optional[T]][int]).replace('typing.', ''),
                         'Callable[..., Union[int, NoneType]]')
        self.assertEqual(repr(Callable[[], List[T]][int]).replace('typing.', ''),
                         'Callable[[], List[int]]')

    def test_generic_forvard_ref(self):
        LLT = List[List['CC']]
        class CC: pass
        self.assertEqual(typing._eval_type(LLT, globals(), locals()), List[List[CC]])
        T = TypeVar('T')
        AT = Tuple[T, ...]
        self.assertIs(typing._eval_type(AT, globals(), locals()), AT)
        CT = Callable[..., List[T]]
        self.assertIs(typing._eval_type(CT, globals(), locals()), CT)

    def test_extended_generic_rules_subclassing(self):
        class T1(Tuple[T, KT]): pass
        class T2(Tuple[T, ...]): pass
        class C1(Callable[[T], T]): pass
        class C2(Callable[..., int]):
            def __call__(self):
                return None

        self.assertEqual(T1.__parameters__, (T, KT))
        self.assertEqual(T1[int, str].__args__, (int, str))
        self.assertEqual(T1[int, T].__origin__, T1)

        self.assertEqual(T2.__parameters__, (T,))
        with self.assertRaises(TypeError):
            T1[int]
        with self.assertRaises(TypeError):
            T2[int, str]

        self.assertEqual(repr(C1[int]).split('.')[-1], 'C1[int]')
        self.assertEqual(C2.__parameters__, ())
        self.assertIsInstance(C2(), collections.Callable)
        self.assertIsSubclass(C2, collections.Callable)
        self.assertIsSubclass(C1, collections.Callable)
        self.assertIsInstance(T1(), tuple)
        self.assertIsSubclass(T2, tuple)
        self.assertIsSubclass(Tuple[int, ...], typing.Sequence)
        self.assertIsSubclass(Tuple[int, ...], typing.Iterable)

    def test_fail_with_bare_union(self):
        with self.assertRaises(TypeError):
            List[Union]
        with self.assertRaises(TypeError):
            Tuple[Optional]
        with self.assertRaises(TypeError):
            ClassVar[ClassVar]
        with self.assertRaises(TypeError):
            List[ClassVar[int]]

    def test_fail_with_bare_generic(self):
        T = TypeVar('T')
        with self.assertRaises(TypeError):
            List[Generic]
        with self.assertRaises(TypeError):
            Tuple[Generic[T]]
        with self.assertRaises(TypeError):
            List[typing.Protocol]
        with self.assertRaises(TypeError):
            isinstance(1, Generic)

    def test_type_erasure_special(self):
        T = TypeVar('T')
        # this is the only test that checks type caching
        self.clear_caches()
        class MyTup(Tuple[T, T]): pass
        self.assertIs(MyTup[int]().__class__, MyTup)
        self.assertIs(MyTup[int]().__orig_class__, MyTup[int])
        class MyCall(Callable[..., T]):
            def __call__(self): return None
        self.assertIs(MyCall[T]().__class__, MyCall)
        self.assertIs(MyCall[T]().__orig_class__, MyCall[T])
        class MyDict(typing.Dict[T, T]): pass
        self.assertIs(MyDict[int]().__class__, MyDict)
        self.assertIs(MyDict[int]().__orig_class__, MyDict[int])
        class MyDef(typing.DefaultDict[str, T]): pass
        self.assertIs(MyDef[int]().__class__, MyDef)
        self.assertIs(MyDef[int]().__orig_class__, MyDef[int])

    def test_all_repr_eq_any(self):
        objs = (getattr(typing, el) for el in typing.__all__)
        for obj in objs:
            self.assertNotEqual(repr(obj), '')
            self.assertEqual(obj, obj)
            if getattr(obj, '__parameters__', None) and len(obj.__parameters__) == 1:
                self.assertEqual(obj[Any].__args__, (Any,))
            if isinstance(obj, type):
                for base in obj.__mro__:
                    self.assertNotEqual(repr(base), '')
                    self.assertEqual(base, base)

    def test_pickle(self):
        global C  # pickle wants to reference the class by name
        T = TypeVar('T')

        class B(Generic[T]):
            pass

        class C(B[int]):
            pass

        c = C()
        c.foo = 42
        c.bar = 'abc'
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z = pickle.dumps(c, proto)
            x = pickle.loads(z)
            self.assertEqual(x.foo, 42)
            self.assertEqual(x.bar, 'abc')
            self.assertEqual(x.__dict__, {'foo': 42, 'bar': 'abc'})
        simples = [Any, Union, Tuple, Callable, ClassVar, List, typing.Iterable]
        for s in simples:
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                z = pickle.dumps(s, proto)
                x = pickle.loads(z)
                self.assertEqual(s, x)

    def test_copy_and_deepcopy(self):
        T = TypeVar('T')
        class Node(Generic[T]): pass
        things = [
            Any,
            Callable[..., T],
            Callable[[int], int],
            ClassVar[List[T]],
            ClassVar[int],
            List['T'],
            Node[Any],
            Node[T],
            Node[int],
            Tuple['T', 'T'],
            Tuple[Any, Any],
            Tuple[T, int],
            Union['T', int],
            Union[T, int],
            typing.Dict[T, Any],
            typing.Dict[int, str],
            typing.Iterable[Any],
            typing.Iterable[T],
            typing.Iterable[int],
            typing.Mapping['T', int]
        ]
        for t in things:
            self.assertEqual(t, deepcopy(t))
            self.assertEqual(t, copy(t))

    def test_copy_generic_instances(self):
        T = TypeVar('T')
        class C(Generic[T]):
            def __init__(self, attr):
                self.attr = attr

        c = C(42)
        self.assertEqual(copy(c).attr, 42)
        self.assertEqual(deepcopy(c).attr, 42)
        self.assertIsNot(copy(c), c)
        self.assertIsNot(deepcopy(c), c)
        c.attr = 1
        self.assertEqual(copy(c).attr, 1)
        self.assertEqual(deepcopy(c).attr, 1)
        ci = C[int](42)
        self.assertEqual(copy(ci).attr, 42)
        self.assertEqual(deepcopy(ci).attr, 42)
        self.assertIsNot(copy(ci), ci)
        self.assertIsNot(deepcopy(ci), ci)
        ci.attr = 1
        self.assertEqual(copy(ci).attr, 1)
        self.assertEqual(deepcopy(ci).attr, 1)
        self.assertEqual(ci.__orig_class__, C[int])

    def test_weakref_all(self):
        T = TypeVar('T')
        things = [Any, Union[T, int], Callable[..., T], Tuple[Any, Any],
                  Optional[List[int]], typing.Mapping[int, str],
                  typing.re.Match[bytes], typing.Iterable['whatever']]
        for t in things:
            self.assertEqual(weakref.ref(t)(), t)

    def test_parameterized_slots(self):
        T = TypeVar('T')
        class C(Generic[T]):
            __slots__ = ('potato',)

        c = C()
        c_int = C[int]()
        self.assertEqual(C.__slots__, C[str].__slots__)

        c.potato = 0
        c_int.potato = 0
        with self.assertRaises(AttributeError):
            c.tomato = 0
        with self.assertRaises(AttributeError):
            c_int.tomato = 0

        self.assertEqual(typing._eval_type(C['C'], globals(), locals()), C[C])
        self.assertEqual(typing._eval_type(C['C'], globals(), locals()).__slots__,
                         C.__slots__)
        self.assertEqual(copy(C[int]), deepcopy(C[int]))

    def test_parameterized_slots_dict(self):
        T = TypeVar('T')
        class D(Generic[T]):
            __slots__ = {'banana': 42}

        d = D()
        d_int = D[int]()
        self.assertEqual(D.__slots__, D[str].__slots__)

        d.banana = 'yes'
        d_int.banana = 'yes'
        with self.assertRaises(AttributeError):
            d.foobar = 'no'
        with self.assertRaises(AttributeError):
            d_int.foobar = 'no'

    def test_errors(self):
        with self.assertRaises(TypeError):
            B = SimpleMapping[XK, Any]

            class C(Generic[B]):
                pass

    def test_repr_2(self):
        PY32 = sys.version_info[:2] < (3, 3)

        class C(Generic[T]):
            pass

        self.assertEqual(C.__module__, __name__)
        if not PY32:
            self.assertEqual(C.__qualname__,
                             'GenericTests.test_repr_2.<locals>.C')
        self.assertEqual(repr(C).split('.')[-1], 'C')
        X = C[int]
        self.assertEqual(X.__module__, __name__)
        if not PY32:
            self.assertTrue(X.__qualname__.endswith('.<locals>.C'))
        self.assertEqual(repr(X).split('.')[-1], 'C[int]')

        class Y(C[int]):
            pass

        self.assertEqual(Y.__module__, __name__)
        if not PY32:
            self.assertEqual(Y.__qualname__,
                             'GenericTests.test_repr_2.<locals>.Y')
        self.assertEqual(repr(Y).split('.')[-1], 'Y')

    def test_eq_1(self):
        self.assertEqual(Generic, Generic)
        self.assertEqual(Generic[T], Generic[T])
        self.assertNotEqual(Generic[KT], Generic[VT])

    def test_eq_2(self):

        class A(Generic[T]):
            pass

        class B(Generic[T]):
            pass

        self.assertEqual(A, A)
        self.assertNotEqual(A, B)
        self.assertEqual(A[T], A[T])
        self.assertNotEqual(A[T], B[T])

    def test_multiple_inheritance(self):

        class A(Generic[T, VT]):
            pass

        class B(Generic[KT, T]):
            pass

        class C(A[T, VT], Generic[VT, T, KT], B[KT, T]):
            pass

        self.assertEqual(C.__parameters__, (VT, T, KT))

    def test_nested(self):

        G = Generic

        class Visitor(G[T]):

            a = None

            def set(self, a):
                self.a = a

            def get(self):
                return self.a

            def visit(self):
                return self.a

        V = Visitor[typing.List[int]]

        class IntListVisitor(V):

            def append(self, x):
                self.a.append(x)

        a = IntListVisitor()
        a.set([])
        a.append(1)
        a.append(42)
        self.assertEqual(a.get(), [1, 42])

    def test_type_erasure(self):
        T = TypeVar('T')

        class Node(Generic[T]):
            def __init__(self, label,
                         left=None,
                         right=None):
                self.label = label  # type: T
                self.left = left  # type: Optional[Node[T]]
                self.right = right  # type: Optional[Node[T]]

        def foo(x):
            a = Node(x)
            b = Node[T](x)
            c = Node[Any](x)
            self.assertIs(type(a), Node)
            self.assertIs(type(b), Node)
            self.assertIs(type(c), Node)
            self.assertEqual(a.label, x)
            self.assertEqual(b.label, x)
            self.assertEqual(c.label, x)

        foo(42)

    def test_implicit_any(self):
        T = TypeVar('T')

        class C(Generic[T]):
            pass

        class D(C):
            pass

        self.assertEqual(D.__parameters__, ())

        with self.assertRaises(Exception):
            D[int]
        with self.assertRaises(Exception):
            D[Any]
        with self.assertRaises(Exception):
            D[T]

    def test_new_with_args(self):

        class A(Generic[T]):
            pass

        class B(object):
            def __new__(cls, arg):
                # call object.__new__
                obj = super(B, cls).__new__(cls)
                obj.arg = arg
                return obj

        # mro: C, A, Generic, B, object
        class C(A, B):
            pass

        c = C('foo')
        self.assertEqual(c.arg, 'foo')

    def test_new_with_args2(self):

        class A(object):
            def __init__(self, arg):
                self.from_a = arg
                # call object
                super(A, self).__init__()

        # mro: C, Generic, A, object
        class C(Generic[T], A):
            def __init__(self, arg):
                self.from_c = arg
                # call Generic
                super(C, self).__init__(arg)

        c = C('foo')
        self.assertEqual(c.from_a, 'foo')
        self.assertEqual(c.from_c, 'foo')

    def test_new_no_args(self):

        class A(Generic[T]):
            pass

        with self.assertRaises(TypeError):
            A('foo')

        class B(object):
            def __new__(cls):
                # call object
                obj = super(B, cls).__new__(cls)
                obj.from_b = 'b'
                return obj

        # mro: C, A, Generic, B, object
        class C(A, B):
            def __init__(self, arg):
                self.arg = arg

            def __new__(cls, arg):
                # call A
                obj = super(C, cls).__new__(cls)
                obj.from_c = 'c'
                return obj

        c = C('foo')
        self.assertEqual(c.arg, 'foo')
        self.assertEqual(c.from_b, 'b')
        self.assertEqual(c.from_c, 'c')


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
            type(ClassVar[Optional[int]])()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, ClassVar[int])
        with self.assertRaises(TypeError):
            issubclass(int, ClassVar)


class FinalTests(BaseTestCase):

    def test_basics(self):
        with self.assertRaises(TypeError):
            Final[1]
        with self.assertRaises(TypeError):
            Final[int, str]
        with self.assertRaises(TypeError):
            Final[int][str]

    def test_repr(self):
        self.assertEqual(repr(Final), 'typing.Final')
        cv = Final[int]
        self.assertEqual(repr(cv), 'typing.Final[int]')
        cv = Final[Employee]
        self.assertEqual(repr(cv), 'typing.Final[%s.Employee]' % __name__)

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(Final)):
                pass
        with self.assertRaises(TypeError):
            class C(type(Final[int])):
                pass

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            Final()
        with self.assertRaises(TypeError):
            type(Final)()
        with self.assertRaises(TypeError):
            type(Final[typing.Optional[int]])()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, Final[int])
        with self.assertRaises(TypeError):
            issubclass(int, Final)


class LiteralTests(BaseTestCase):
    def test_basics(self):
        Literal[1]
        Literal[1, 2, 3]
        Literal["x", "y", "z"]
        Literal[None]

    def test_illegal_parameters_do_not_raise_runtime_errors(self):
        # Type checkers should reject these types, but we do not
        # raise errors at runtime to maintain maximium flexibility
        Literal[int]
        Literal[Literal[1, 2], Literal[4, 5]]
        Literal[3j + 2, ..., ()]
        Literal[b"foo", u"bar"]
        Literal[{"foo": 3, "bar": 4}]
        Literal[T]

    def test_literals_inside_other_types(self):
        typing.List[Literal[1, 2, 3]]
        typing.List[Literal[("foo", "bar", "baz")]]

    def test_repr(self):
        self.assertEqual(repr(Literal[1]), "typing.Literal[1]")
        self.assertEqual(repr(Literal[1, True, "foo"]), "typing.Literal[1, True, u'foo']")
        self.assertEqual(repr(Literal[int]), "typing.Literal[int]")
        self.assertEqual(repr(Literal), "typing.Literal")
        self.assertEqual(repr(Literal[None]), "typing.Literal[None]")

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            Literal()
        with self.assertRaises(TypeError):
            Literal[1]()
        with self.assertRaises(TypeError):
            type(Literal)()
        with self.assertRaises(TypeError):
            type(Literal[1])()

    def test_no_isinstance_or_issubclass(self):
        with self.assertRaises(TypeError):
            isinstance(1, Literal[1])
        with self.assertRaises(TypeError):
            isinstance(int, Literal[1])
        with self.assertRaises(TypeError):
            issubclass(1, Literal[1])
        with self.assertRaises(TypeError):
            issubclass(int, Literal[1])

    def test_no_subclassing(self):
        with self.assertRaises(TypeError):
            class Foo(Literal[1]): pass
        with self.assertRaises(TypeError):
            class Bar(Literal): pass

    def test_no_multiple_subscripts(self):
        with self.assertRaises(TypeError):
            Literal[1][1]


class CastTests(BaseTestCase):

    def test_basics(self):
        self.assertEqual(cast(int, 42), 42)
        self.assertEqual(cast(float, 42), 42)
        self.assertIs(type(cast(float, 42)), int)
        self.assertEqual(cast(Any, 42), 42)
        self.assertEqual(cast(list, 42), 42)
        self.assertEqual(cast(Union[str, float], 42), 42)
        self.assertEqual(cast(AnyStr, 42), 42)
        self.assertEqual(cast(None, 42), 42)

    def test_errors(self):
        # Bogus calls are not expected to fail.
        cast(42, 42)
        cast('hello', 42)


class ForwardRefTests(BaseTestCase):

    def test_forwardref_instance_type_error(self):
        fr = typing._ForwardRef('int')
        with self.assertRaises(TypeError):
            isinstance(42, fr)

    def test_syntax_error(self):

        with self.assertRaises(SyntaxError):
            Generic['/T']

    def test_forwardref_subclass_type_error(self):
        fr = typing._ForwardRef('int')
        with self.assertRaises(TypeError):
            issubclass(int, fr)

    def test_forward_equality(self):
        fr = typing._ForwardRef('int')
        self.assertEqual(fr, typing._ForwardRef('int'))
        self.assertNotEqual(List['int'], List[int])

    def test_forward_repr(self):
        self.assertEqual(repr(List['int']), "typing.List[_ForwardRef(%r)]" % 'int')


class OverloadTests(BaseTestCase):

    def test_overload_fails(self):
        from typing import overload

        with self.assertRaises(RuntimeError):

            @overload
            def blah():
                pass

            blah()

    def test_overload_succeeds(self):
        from typing import overload

        @overload
        def blah():
            pass

        def blah():
            pass

        blah()


class CollectionsAbcTests(BaseTestCase):

    def test_hashable(self):
        self.assertIsInstance(42, typing.Hashable)
        self.assertNotIsInstance([], typing.Hashable)

    def test_iterable(self):
        self.assertIsInstance([], typing.Iterable)
        # Due to ABC caching, the second time takes a separate code
        # path and could fail.  So call this a few times.
        self.assertIsInstance([], typing.Iterable)
        self.assertIsInstance([], typing.Iterable)
        self.assertNotIsInstance(42, typing.Iterable)
        # Just in case, also test issubclass() a few times.
        self.assertIsSubclass(list, typing.Iterable)
        self.assertIsSubclass(list, typing.Iterable)

    def test_iterator(self):
        it = iter([])
        self.assertIsInstance(it, typing.Iterator)
        self.assertNotIsInstance(42, typing.Iterator)

    def test_sized(self):
        self.assertIsInstance([], typing.Sized)
        self.assertNotIsInstance(42, typing.Sized)

    def test_container(self):
        self.assertIsInstance([], typing.Container)
        self.assertNotIsInstance(42, typing.Container)

    def test_abstractset(self):
        self.assertIsInstance(set(), typing.AbstractSet)
        self.assertNotIsInstance(42, typing.AbstractSet)

    def test_mutableset(self):
        self.assertIsInstance(set(), typing.MutableSet)
        self.assertNotIsInstance(frozenset(), typing.MutableSet)

    def test_mapping(self):
        self.assertIsInstance({}, typing.Mapping)
        self.assertNotIsInstance(42, typing.Mapping)

    def test_mutablemapping(self):
        self.assertIsInstance({}, typing.MutableMapping)
        self.assertNotIsInstance(42, typing.MutableMapping)

    def test_sequence(self):
        self.assertIsInstance([], typing.Sequence)
        self.assertNotIsInstance(42, typing.Sequence)

    def test_mutablesequence(self):
        self.assertIsInstance([], typing.MutableSequence)
        self.assertNotIsInstance((), typing.MutableSequence)

    def test_bytestring(self):
        self.assertIsInstance(b'', typing.ByteString)
        self.assertIsInstance(bytearray(b''), typing.ByteString)

    def test_list(self):
        self.assertIsSubclass(list, typing.List)

    def test_deque(self):
        self.assertIsSubclass(collections.deque, typing.Deque)
        class MyDeque(typing.Deque[int]): pass
        self.assertIsInstance(MyDeque(), collections.deque)

    def test_counter(self):
        self.assertIsSubclass(collections.Counter, typing.Counter)

    def test_set(self):
        self.assertIsSubclass(set, typing.Set)
        self.assertNotIsSubclass(frozenset, typing.Set)

    def test_frozenset(self):
        self.assertIsSubclass(frozenset, typing.FrozenSet)
        self.assertNotIsSubclass(set, typing.FrozenSet)

    def test_dict(self):
        self.assertIsSubclass(dict, typing.Dict)

    def test_no_list_instantiation(self):
        with self.assertRaises(TypeError):
            typing.List()
        with self.assertRaises(TypeError):
            typing.List[T]()
        with self.assertRaises(TypeError):
            typing.List[int]()

    def test_list_subclass(self):

        class MyList(typing.List[int]):
            pass

        a = MyList()
        self.assertIsInstance(a, MyList)
        self.assertIsInstance(a, typing.Sequence)

        self.assertIsSubclass(MyList, list)
        self.assertNotIsSubclass(list, MyList)

    def test_no_dict_instantiation(self):
        with self.assertRaises(TypeError):
            typing.Dict()
        with self.assertRaises(TypeError):
            typing.Dict[KT, VT]()
        with self.assertRaises(TypeError):
            typing.Dict[str, int]()

    def test_dict_subclass(self):

        class MyDict(typing.Dict[str, int]):
            pass

        d = MyDict()
        self.assertIsInstance(d, MyDict)
        self.assertIsInstance(d, typing.MutableMapping)

        self.assertIsSubclass(MyDict, dict)
        self.assertNotIsSubclass(dict, MyDict)

    def test_defaultdict_instantiation(self):
        self.assertIs(type(typing.DefaultDict()), collections.defaultdict)
        self.assertIs(type(typing.DefaultDict[KT, VT]()), collections.defaultdict)
        self.assertIs(type(typing.DefaultDict[str, int]()), collections.defaultdict)

    def test_defaultdict_subclass(self):

        class MyDefDict(typing.DefaultDict[str, int]):
            pass

        dd = MyDefDict()
        self.assertIsInstance(dd, MyDefDict)

        self.assertIsSubclass(MyDefDict, collections.defaultdict)
        self.assertNotIsSubclass(collections.defaultdict, MyDefDict)

    def test_deque_instantiation(self):
        self.assertIs(type(typing.Deque()), collections.deque)
        self.assertIs(type(typing.Deque[T]()), collections.deque)
        self.assertIs(type(typing.Deque[int]()), collections.deque)
        class D(typing.Deque[T]): pass
        self.assertIs(type(D[int]()), D)

    def test_counter_instantiation(self):
        self.assertIs(type(typing.Counter()), collections.Counter)
        self.assertIs(type(typing.Counter[T]()), collections.Counter)
        self.assertIs(type(typing.Counter[int]()), collections.Counter)
        class C(typing.Counter[T]): pass
        self.assertIs(type(C[int]()), C)

    def test_counter_subclass_instantiation(self):

        class MyCounter(typing.Counter[int]):
            pass

        d = MyCounter()
        self.assertIsInstance(d, MyCounter)
        self.assertIsInstance(d, typing.Counter)
        self.assertIsInstance(d, collections.Counter)

    def test_no_set_instantiation(self):
        with self.assertRaises(TypeError):
            typing.Set()
        with self.assertRaises(TypeError):
            typing.Set[T]()
        with self.assertRaises(TypeError):
            typing.Set[int]()

    def test_set_subclass_instantiation(self):

        class MySet(typing.Set[int]):
            pass

        d = MySet()
        self.assertIsInstance(d, MySet)

    def test_no_frozenset_instantiation(self):
        with self.assertRaises(TypeError):
            typing.FrozenSet()
        with self.assertRaises(TypeError):
            typing.FrozenSet[T]()
        with self.assertRaises(TypeError):
            typing.FrozenSet[int]()

    def test_frozenset_subclass_instantiation(self):

        class MyFrozenSet(typing.FrozenSet[int]):
            pass

        d = MyFrozenSet()
        self.assertIsInstance(d, MyFrozenSet)

    def test_no_tuple_instantiation(self):
        with self.assertRaises(TypeError):
            Tuple()
        with self.assertRaises(TypeError):
            Tuple[T]()
        with self.assertRaises(TypeError):
            Tuple[int]()

    def test_generator(self):
        def foo():
            yield 42
        g = foo()
        self.assertIsSubclass(type(g), typing.Generator)

    def test_no_generator_instantiation(self):
        with self.assertRaises(TypeError):
            typing.Generator()
        with self.assertRaises(TypeError):
            typing.Generator[T, T, T]()
        with self.assertRaises(TypeError):
            typing.Generator[int, int, int]()

    def test_subclassing(self):

        class MMA(typing.MutableMapping):
            pass

        with self.assertRaises(TypeError):  # It's abstract
            MMA()

        class MMC(MMA):
            def __getitem__(self, k):
                return None
            def __setitem__(self, k, v):
                pass
            def __delitem__(self, k):
                pass
            def __iter__(self):
                return iter(())
            def __len__(self):
                return 0

        self.assertEqual(len(MMC()), 0)
        assert callable(MMC.update)
        self.assertIsInstance(MMC(), typing.Mapping)

        class MMB(typing.MutableMapping[KT, VT]):
            def __getitem__(self, k):
                return None
            def __setitem__(self, k, v):
                pass
            def __delitem__(self, k):
                pass
            def __iter__(self):
                return iter(())
            def __len__(self):
                return 0

        self.assertEqual(len(MMB()), 0)
        self.assertEqual(len(MMB[str, str]()), 0)
        self.assertEqual(len(MMB[KT, VT]()), 0)

        self.assertNotIsSubclass(dict, MMA)
        self.assertNotIsSubclass(dict, MMB)

        self.assertIsSubclass(MMA, typing.Mapping)
        self.assertIsSubclass(MMB, typing.Mapping)
        self.assertIsSubclass(MMC, typing.Mapping)

        self.assertIsInstance(MMB[KT, VT](), typing.Mapping)
        self.assertIsInstance(MMB[KT, VT](), collections.Mapping)

        self.assertIsSubclass(MMA, collections.Mapping)
        self.assertIsSubclass(MMB, collections.Mapping)
        self.assertIsSubclass(MMC, collections.Mapping)

        self.assertIsSubclass(MMB[str, str], typing.Mapping)
        self.assertIsSubclass(MMC, MMA)

        class It(typing.Iterable): pass
        self.assertNotIsSubclass(list, It)

        class G(typing.Generator[int, int, int]): pass
        def g(): yield 0
        self.assertIsSubclass(G, typing.Generator)
        self.assertIsSubclass(G, typing.Iterable)
        if hasattr(collections, 'Generator'):
            self.assertIsSubclass(G, collections.Generator)
        self.assertIsSubclass(G, collections.Iterable)
        self.assertNotIsSubclass(type(g), G)

    def test_subclassing_subclasshook(self):

        class Base(typing.Iterable):
            @classmethod
            def __subclasshook__(cls, other):
                if other.__name__ == 'Foo':
                    return True
                else:
                    return False

        class C(Base): pass
        class Foo: pass
        class Bar: pass
        self.assertIsSubclass(Foo, Base)
        self.assertIsSubclass(Foo, C)
        self.assertNotIsSubclass(Bar, C)

    def test_subclassing_register(self):

        class A(typing.Container): pass
        class B(A): pass

        class C: pass
        A.register(C)
        self.assertIsSubclass(C, A)
        self.assertNotIsSubclass(C, B)

        class D: pass
        B.register(D)
        self.assertIsSubclass(D, A)
        self.assertIsSubclass(D, B)

        class M(): pass
        collections.MutableMapping.register(M)
        self.assertIsSubclass(M, typing.Mapping)

    def test_collections_as_base(self):

        class M(collections.Mapping): pass
        self.assertIsSubclass(M, typing.Mapping)
        self.assertIsSubclass(M, typing.Iterable)

        class S(collections.MutableSequence): pass
        self.assertIsSubclass(S, typing.MutableSequence)
        self.assertIsSubclass(S, typing.Iterable)

        class It(collections.Iterable): pass
        self.assertIsSubclass(It, typing.Iterable)

        class A(collections.Mapping): pass
        class B: pass
        A.register(B)
        self.assertIsSubclass(B, typing.Mapping)


class OtherABCTests(BaseTestCase):

    def test_contextmanager(self):
        @contextlib.contextmanager
        def manager():
            yield 42

        cm = manager()
        self.assertIsInstance(cm, typing.ContextManager)
        self.assertNotIsInstance(42, typing.ContextManager)


class TypeTests(BaseTestCase):

    def test_type_basic(self):

        class User(object): pass
        class BasicUser(User): pass
        class ProUser(User): pass

        def new_user(user_class):
            # type: (Type[User]) -> User
            return user_class()

        new_user(BasicUser)

    def test_type_typevar(self):

        class User(object): pass
        class BasicUser(User): pass
        class ProUser(User): pass

        global U
        U = TypeVar('U', bound=User)

        def new_user(user_class):
            # type: (Type[U]) -> U
            return user_class()

        new_user(BasicUser)

    def test_type_optional(self):
        A = Optional[Type[BaseException]]  # noqa

        def foo(a):
            # type: (A) -> Optional[BaseException]
            if a is None:
                return None
            else:
                return a()

        assert isinstance(foo(KeyboardInterrupt), KeyboardInterrupt)
        assert foo(None) is None


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


class NamedTupleTests(BaseTestCase):

    def test_basics(self):
        Emp = NamedTuple('Emp', [('name', str), ('id', int)])
        self.assertIsSubclass(Emp, tuple)
        joe = Emp('Joe', 42)
        jim = Emp(name='Jim', id=1)
        self.assertIsInstance(joe, Emp)
        self.assertIsInstance(joe, tuple)
        self.assertEqual(joe.name, 'Joe')
        self.assertEqual(joe.id, 42)
        self.assertEqual(jim.name, 'Jim')
        self.assertEqual(jim.id, 1)
        self.assertEqual(Emp.__name__, 'Emp')
        self.assertEqual(Emp._fields, ('name', 'id'))
        self.assertEqual(Emp._field_types, dict(name=str, id=int))

    def test_pickle(self):
        global Emp  # pickle wants to reference the class by name
        Emp = NamedTuple('Emp', [('name', str), ('id', int)])
        jane = Emp('jane', 37)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z = pickle.dumps(jane, proto)
            jane2 = pickle.loads(z)
            self.assertEqual(jane2, jane)


class TypedDictTests(BaseTestCase):

    def test_basics_iterable_syntax(self):
        Emp = TypedDict(b'Emp', {'name': str, 'id': int})
        self.assertIsSubclass(Emp, dict)
        self.assertIsSubclass(Emp, typing.MutableMapping)
        if sys.version_info[0] >= 3:
            import collections.abc
            self.assertNotIsSubclass(Emp, collections.abc.Sequence)
        jim = Emp(name='Jim', id=1)
        self.assertIs(type(jim), dict)
        self.assertEqual(jim['name'], 'Jim')
        self.assertEqual(jim['id'], 1)
        self.assertEqual(Emp.__name__, 'Emp')
        self.assertEqual(Emp.__module__, __name__)
        self.assertEqual(Emp.__bases__, (dict,))
        self.assertEqual(Emp.__annotations__, {'name': str, 'id': int})
        self.assertEqual(Emp.__total__, True)

    def test_basics_keywords_syntax(self):
        Emp = TypedDict(b'Emp', name=str, id=int)
        self.assertIsSubclass(Emp, dict)
        self.assertIsSubclass(Emp, typing.MutableMapping)
        if sys.version_info[0] >= 3:
            import collections.abc
            self.assertNotIsSubclass(Emp, collections.abc.Sequence)
        jim = Emp(name='Jim', id=1)
        self.assertIs(type(jim), dict)
        self.assertEqual(jim['name'], 'Jim')
        self.assertEqual(jim['id'], 1)
        self.assertEqual(Emp.__name__, 'Emp')
        self.assertEqual(Emp.__module__, __name__)
        self.assertEqual(Emp.__bases__, (dict,))
        self.assertEqual(Emp.__annotations__, {'name': str, 'id': int})
        self.assertEqual(Emp.__total__, True)

    def test_typeddict_errors(self):
        Emp = TypedDict(b'Emp', {'name': str, 'id': int})
        self.assertEqual(TypedDict.__module__, 'typing')
        jim = Emp(name='Jim', id=1)
        with self.assertRaises(TypeError):
            isinstance({}, Emp)
        with self.assertRaises(TypeError):
            isinstance(jim, Emp)
        with self.assertRaises(TypeError):
            issubclass(dict, Emp)
        with self.assertRaises(TypeError):
            TypedDict('Hi', x=1)
        with self.assertRaises(TypeError):
            TypedDict('Hi', [('x', int), ('y', 1)])
        with self.assertRaises(TypeError):
            TypedDict('Hi', [('x', int)], y=int)

    def test_pickle(self):
        global EmpD  # pickle wants to reference the class by name
        EmpD = TypedDict(b'EmpD', name=str, id=int)
        jane = EmpD({'name': 'jane', 'id': 37})
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z = pickle.dumps(jane, proto)
            jane2 = pickle.loads(z)
            self.assertEqual(jane2, jane)
            self.assertEqual(jane2, {'name': 'jane', 'id': 37})
            ZZ = pickle.dumps(EmpD, proto)
            EmpDnew = pickle.loads(ZZ)
            self.assertEqual(EmpDnew({'name': 'jane', 'id': 37}), jane)

    def test_optional(self):
        EmpD = TypedDict(b'EmpD', name=str, id=int)

        self.assertEqual(typing.Optional[EmpD], typing.Union[None, EmpD])
        self.assertNotEqual(typing.List[EmpD], typing.Tuple[EmpD])

    def test_total(self):
        D = TypedDict(b'D', {'x': int}, total=False)
        self.assertEqual(D(), {})
        self.assertEqual(D(x=1), {'x': 1})
        self.assertEqual(D.__total__, False)


class IOTests(BaseTestCase):

    def test_io_submodule(self):
        from typing.io import IO, TextIO, BinaryIO, __all__, __name__
        self.assertIs(IO, typing.IO)
        self.assertIs(TextIO, typing.TextIO)
        self.assertIs(BinaryIO, typing.BinaryIO)
        self.assertEqual(set(__all__), set(['IO', 'TextIO', 'BinaryIO']))
        self.assertEqual(__name__, 'typing.io')


class RETests(BaseTestCase):
    # Much of this is really testing _TypeAlias.

    def test_basics(self):
        pat = re.compile('[a-z]+', re.I)
        self.assertIsSubclass(pat.__class__, Pattern)
        self.assertIsSubclass(type(pat), Pattern)
        self.assertIsInstance(pat, Pattern)

        mat = pat.search('12345abcde.....')
        self.assertIsSubclass(mat.__class__, Match)
        self.assertIsSubclass(type(mat), Match)
        self.assertIsInstance(mat, Match)

        # these should just work
        Pattern[Union[str, bytes]]
        Match[Union[bytes, str]]

    def test_alias_equality(self):
        self.assertEqual(Pattern[str], Pattern[str])
        self.assertNotEqual(Pattern[str], Pattern[bytes])
        self.assertNotEqual(Pattern[str], Match[str])
        self.assertNotEqual(Pattern[str], str)

    def test_errors(self):
        with self.assertRaises(TypeError):
            # Doesn't fit AnyStr.
            Pattern[int]
        with self.assertRaises(TypeError):
            # Can't change type vars?
            Match[T]
        m = Match[Union[str, bytes]]
        with self.assertRaises(TypeError):
            # Too complicated?
            m[str]
        with self.assertRaises(TypeError):
            # We don't support isinstance().
            isinstance(42, Pattern[str])
        with self.assertRaises(TypeError):
            # We don't support issubclass().
            issubclass(Pattern[bytes], Pattern[str])

    def test_repr(self):
        self.assertEqual(repr(Pattern), 'Pattern[~AnyStr]')
        self.assertEqual(repr(Pattern[unicode]), 'Pattern[unicode]')
        self.assertEqual(repr(Pattern[str]), 'Pattern[str]')
        self.assertEqual(repr(Match), 'Match[~AnyStr]')
        self.assertEqual(repr(Match[unicode]), 'Match[unicode]')
        self.assertEqual(repr(Match[str]), 'Match[str]')

    def test_re_submodule(self):
        from typing.re import Match, Pattern, __all__, __name__
        self.assertIs(Match, typing.Match)
        self.assertIs(Pattern, typing.Pattern)
        self.assertEqual(set(__all__), set(['Match', 'Pattern']))
        self.assertEqual(__name__, 'typing.re')

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError) as ex:

            class A(typing.Match):
                pass

        self.assertEqual(str(ex.exception),
                         "Cannot subclass typing._TypeAlias")


class AllTests(BaseTestCase):
    """Tests for __all__."""

    def test_all(self):
        from typing import __all__ as a
        # Just spot-check the first and last of every category.
        self.assertIn('AbstractSet', a)
        self.assertIn('ValuesView', a)
        self.assertIn('cast', a)
        self.assertIn('overload', a)
        # Check that io and re are not exported.
        self.assertNotIn('io', a)
        self.assertNotIn('re', a)
        # Spot-check that stdlib modules aren't exported.
        self.assertNotIn('os', a)
        self.assertNotIn('sys', a)
        # Check that Text is defined.
        self.assertIn('Text', a)
        # Check previously missing class.
        self.assertIn('SupportsComplex', a)

    def test_respect_no_type_check(self):
        @typing.no_type_check
        class NoTpCheck(object):
            class Inn(object):
                def __init__(self, x):
                    # type: (this is not actually a type) -> None  # noqa
                    pass
        self.assertTrue(NoTpCheck.__no_type_check__)
        self.assertTrue(NoTpCheck.Inn.__init__.__no_type_check__)

    def test_get_type_hints_dummy(self):

        def foo(x):
            # type: (int) -> int
            return x + 1

        self.assertIsNone(typing.get_type_hints(foo))

    # def test_typing_compiles_with_opt(self):
    #     file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                              'typing.py')
    #     try:
    #         subprocess.check_output([sys.executable, '-OO', file_path],
    #                                 stderr=subprocess.STDOUT)
    #     except subprocess.CalledProcessError:
    #         self.fail('Module does not compile with optimize=2 (-OO flag).')


if __name__ == '__main__':
    main()
