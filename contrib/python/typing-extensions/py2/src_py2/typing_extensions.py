import abc
import typing
from typing import (  # noqa
    # These are imported for re-export.
    ClassVar, Type, Generic, Callable, GenericMeta, TypingMeta,
    Counter, DefaultDict, Deque, TypeVar, Tuple, Final, final,
    NewType, overload, Text, TYPE_CHECKING, Literal, TypedDict, Protocol,
    SupportsIndex,
    runtime_checkable,
    # We use internal typing helpers here, but this significantly reduces
    # code duplication. (Also this is only until Protocol is in typing.)
    _type_vars, _tp_cache, _type_check,
)

# Please keep __all__ alphabetized within each category.
__all__ = [
    # Super-special typing primitives.
    'ClassVar',
    'Final',
    'Protocol',
    'Type',
    'TypedDict',

    # Concrete collection types.
    'ContextManager',
    'Counter',
    'Deque',
    'DefaultDict',

    # Structural checks, a.k.a. protocols.
    'SupportsIndex',

    # One-off things.
    'final',
    'IntVar',
    'Literal',
    'NewType',
    'overload',
    'runtime_checkable',
    'Text',
    'TYPE_CHECKING',
]


if hasattr(typing, 'NoReturn'):
    NoReturn = typing.NoReturn
else:
    # TODO: Remove once typing.py has been updated
    class _NoReturnMeta(typing.TypingMeta):
        """Metaclass for NoReturn."""

        def __new__(cls, name, bases, namespace):
            cls.assert_no_subclassing(bases)
            self = super(_NoReturnMeta, cls).__new__(cls, name, bases, namespace)
            return self

    class _NoReturn(typing._FinalTypingBase):
        """Special type indicating functions that never return.
        Example::
          from typing import NoReturn
          def stop() -> NoReturn:
              raise Exception('no way')
        This type is invalid in other positions, e.g., ``List[NoReturn]``
        will fail in static type checkers.
        """
        __metaclass__ = _NoReturnMeta
        __slots__ = ()

        def __instancecheck__(self, obj):
            raise TypeError("NoReturn cannot be used with isinstance().")

        def __subclasscheck__(self, cls):
            raise TypeError("NoReturn cannot be used with issubclass().")

    NoReturn = _NoReturn(_root=True)


T_co = typing.TypeVar('T_co', covariant=True)

if hasattr(typing, 'ContextManager'):
    ContextManager = typing.ContextManager
else:
    # TODO: Remove once typing.py has been updated
    class ContextManager(typing.Generic[T_co]):
        __slots__ = ()

        def __enter__(self):
            return self

        @abc.abstractmethod
        def __exit__(self, exc_type, exc_value, traceback):
            return None

        @classmethod
        def __subclasshook__(cls, C):
            if cls is ContextManager:
                # In Python 3.6+, it is possible to set a method to None to
                # explicitly indicate that the class does not implement an ABC
                # (https://bugs.python.org/issue25958), but we do not support
                # that pattern here because this fallback class is only used
                # in Python 3.5 and earlier.
                if (any("__enter__" in B.__dict__ for B in C.__mro__) and
                    any("__exit__" in B.__dict__ for B in C.__mro__)):
                    return True
            return NotImplemented


def IntVar(name):
    return TypeVar(name)


def _is_dunder(name):
    """Returns True if name is a __dunder_variable_name__."""
    return len(name) > 4 and name.startswith('__') and name.endswith('__')


class AnnotatedMeta(GenericMeta):
    """Metaclass for Annotated"""

    def __new__(cls, name, bases, namespace, **kwargs):
        if any(b is not object for b in bases):
            raise TypeError("Cannot subclass %s" % Annotated)
        return super(AnnotatedMeta, cls).__new__(cls, name, bases, namespace, **kwargs)

    @property
    def __metadata__(self):
        return self._subs_tree()[2]

    def _tree_repr(self, tree):
        cls, origin, metadata = tree
        if not isinstance(origin, tuple):
            tp_repr = typing._type_repr(origin)
        else:
            tp_repr = origin[0]._tree_repr(origin)
        metadata_reprs = ", ".join(repr(arg) for arg in metadata)
        return '%s[%s, %s]' % (cls, tp_repr, metadata_reprs)

    def _subs_tree(self, tvars=None, args=None):
        if self is Annotated:
            return Annotated
        res = super(AnnotatedMeta, self)._subs_tree(tvars=tvars, args=args)
        # Flatten nested Annotated
        if isinstance(res[1], tuple) and res[1][0] is Annotated:
            sub_tp = res[1][1]
            sub_annot = res[1][2]
            return (Annotated, sub_tp, sub_annot + res[2])
        return res

    def _get_cons(self):
        """Return the class used to create instance of this type."""
        if self.__origin__ is None:
            raise TypeError("Cannot get the underlying type of a non-specialized "
                            "Annotated type.")
        tree = self._subs_tree()
        while isinstance(tree, tuple) and tree[0] is Annotated:
            tree = tree[1]
        if isinstance(tree, tuple):
            return tree[0]
        else:
            return tree

    @_tp_cache
    def __getitem__(self, params):
        if not isinstance(params, tuple):
            params = (params,)
        if self.__origin__ is not None:  # specializing an instantiated type
            return super(AnnotatedMeta, self).__getitem__(params)
        elif not isinstance(params, tuple) or len(params) < 2:
            raise TypeError("Annotated[...] should be instantiated with at "
                            "least two arguments (a type and an annotation).")
        else:
            msg = "Annotated[t, ...]: t must be a type."
            tp = typing._type_check(params[0], msg)
            metadata = tuple(params[1:])
        return self.__class__(
            self.__name__,
            self.__bases__,
            dict(self.__dict__),
            tvars=_type_vars((tp,)),
            # Metadata is a tuple so it won't be touched by _replace_args et al.
            args=(tp, metadata),
            origin=self,
        )

    def __call__(self, *args, **kwargs):
        cons = self._get_cons()
        result = cons(*args, **kwargs)
        try:
            result.__orig_class__ = self
        except AttributeError:
            pass
        return result

    def __getattr__(self, attr):
        # For simplicity we just don't relay all dunder names
        if self.__origin__ is not None and not _is_dunder(attr):
            return getattr(self._get_cons(), attr)
        raise AttributeError(attr)

    def __setattr__(self, attr, value):
        if _is_dunder(attr) or attr.startswith('_abc_'):
            super(AnnotatedMeta, self).__setattr__(attr, value)
        elif self.__origin__ is None:
            raise AttributeError(attr)
        else:
            setattr(self._get_cons(), attr, value)


class Annotated(object):
    """Add context specific metadata to a type.

    Example: Annotated[int, runtime_check.Unsigned] indicates to the
    hypothetical runtime_check module that this type is an unsigned int.
    Every other consumer of this type can ignore this metadata and treat
    this type as int.

    The first argument to Annotated must be a valid type, the remaining
    arguments are kept as a tuple in the __metadata__ field.

    Details:

    - It's an error to call `Annotated` with less than two arguments.
    - Nested Annotated are flattened::

        Annotated[Annotated[int, Ann1, Ann2], Ann3] == Annotated[int, Ann1, Ann2, Ann3]

    - Instantiating an annotated type is equivalent to instantiating the
    underlying type::

        Annotated[C, Ann1](5) == C(5)

    - Annotated can be used as a generic type alias::

        Optimized = Annotated[T, runtime.Optimize()]
        Optimized[int] == Annotated[int, runtime.Optimize()]

        OptimizedList = Annotated[List[T], runtime.Optimize()]
        OptimizedList[int] == Annotated[List[int], runtime.Optimize()]
    """
    __metaclass__ = AnnotatedMeta
    __slots__ = ()


class _TypeAliasMeta(typing.TypingMeta):
    """Metaclass for TypeAlias"""

    def __new__(cls, name, bases, namespace):
        cls.assert_no_subclassing(bases)
        self = super(_TypeAliasMeta, cls).__new__(cls, name, bases, namespace)
        return self

    def __repr__(self):
        return 'typing_extensions.TypeAlias'


class _TypeAliasBase(typing._FinalTypingBase):
    """Special marker indicating that an assignment should
    be recognized as a proper type alias definition by type
    checkers.

    For example::

        Predicate = Callable[..., bool]  # type: TypeAlias

    It's invalid when used anywhere except as in the example above.
    """
    __metaclass__ = _TypeAliasMeta
    __slots__ = ()

    def __instancecheck__(self, obj):
        raise TypeError("TypeAlias cannot be used with isinstance().")

    def __subclasscheck__(self, cls):
        raise TypeError("TypeAlias cannot be used with issubclass().")

    def __repr__(self):
        return 'typing_extensions.TypeAlias'


TypeAlias = _TypeAliasBase(_root=True)

# This alias exists for backwards compatibility.
runtime = runtime_checkable
