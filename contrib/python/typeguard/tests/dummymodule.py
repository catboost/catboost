"""Module docstring."""
from __future__ import absolute_import, division

from typing import no_type_check, no_type_check_decorator

from typeguard import typeguard_ignore


@no_type_check_decorator
def dummy_decorator(func):
    return func


def type_checked_func(x: int, y: int) -> int:
    return x * y


@no_type_check
def non_type_checked_func(x: int, y: str) -> 6:
    return 'foo'


@dummy_decorator
def non_type_checked_decorated_func(x: int, y: str) -> 6:
    return 'foo'


@typeguard_ignore
def non_typeguard_checked_func(x: int, y: str) -> 6:
    return 'foo'


def dynamic_type_checking_func(arg, argtype, return_annotation):
    def inner(x: argtype) -> return_annotation:
        return str(x)

    return inner(arg)


class Metaclass(type):
    pass


class DummyClass(metaclass=Metaclass):
    def type_checked_method(self, x: int, y: int) -> int:
        return x * y

    @classmethod
    def type_checked_classmethod(cls, x: int, y: int) -> int:
        return x * y

    @staticmethod
    def type_checked_staticmethod(x: int, y: int) -> int:
        return x * y

    @classmethod
    def undocumented_classmethod(cls, x, y):
        pass

    @staticmethod
    def undocumented_staticmethod(x, y):
        pass

    @property
    def unannotated_property(self):
        return None


def outer():
    class Inner:
        pass

    def create_inner() -> 'Inner':
        return Inner()

    return create_inner


class Outer:
    class Inner:
        pass

    def create_inner(self) -> 'Inner':
        return Outer.Inner()

    @classmethod
    def create_inner_classmethod(cls) -> 'Inner':
        return Outer.Inner()

    @staticmethod
    def create_inner_staticmethod() -> 'Inner':
        return Outer.Inner()
