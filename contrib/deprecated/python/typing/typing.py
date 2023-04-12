from __future__ import absolute_import, unicode_literals

import abc
from abc import abstractmethod, abstractproperty
import collections
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import copy
try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc  # Fallback for PY3.2.


# Please keep __all__ alphabetized within each category.
__all__ = [
    # Super-special typing primitives.
    'Any',
    'Callable',
    'ClassVar',
    'Final',
    'Generic',
    'Literal',
    'Optional',
    'Protocol',
    'Tuple',
    'Type',
    'TypeVar',
    'Union',

    # ABCs (from collections.abc).
    'AbstractSet',  # collections.abc.Set.
    'GenericMeta',  # subclass of abc.ABCMeta and a metaclass
                    # for 'Generic' and ABCs below.
    'ByteString',
    'Container',
    'ContextManager',
    'Hashable',
    'ItemsView',
    'Iterable',
    'Iterator',
    'KeysView',
    'Mapping',
    'MappingView',
    'MutableMapping',
    'MutableSequence',
    'MutableSet',
    'Sequence',
    'Sized',
    'ValuesView',

    # Structural checks, a.k.a. protocols.
    'Reversible',
    'SupportsAbs',
    'SupportsComplex',
    'SupportsFloat',
    'SupportsIndex',
    'SupportsInt',

    # Concrete collection types.
    'Counter',
    'Deque',
    'Dict',
    'DefaultDict',
    'List',
    'Set',
    'FrozenSet',
    'NamedTuple',  # Not really a type.
    'TypedDict',  # Not really a type.
    'Generator',

    # One-off things.
    'AnyStr',
    'cast',
    'final',
    'get_type_hints',
    'NewType',
    'no_type_check',
    'no_type_check_decorator',
    'NoReturn',
    'overload',
    'runtime_checkable',
    'Text',
    'TYPE_CHECKING',
]

# The pseudo-submodules 're' and 'io' are part of the public
# namespace, but excluded from __all__ because they might stomp on
# legitimate imports of those modules.


def _qualname(x):
    if sys.version_info[:2] >= (3, 3):
        return x.__qualname__
    else:
        # Fall back to just name.
        return x.__name__


def _trim_name(nm):
    whitelist = ('_TypeAlias', '_ForwardRef', '_TypingBase', '_FinalTypingBase')
    if nm.startswith('_') and nm not in whitelist:
        nm = nm[1:]
    return nm


class TypingMeta(type):
    """Metaclass for most types defined in typing module
    (not a part of public API).

    This also defines a dummy constructor (all the work for most typing
    constructs is done in __new__) and a nicer repr().
    """

    _is_protocol = False

    def __new__(cls, name, bases, namespace):
        return super(TypingMeta, cls).__new__(cls, str(name), bases, namespace)

    @classmethod
    def assert_no_subclassing(cls, bases):
        for base in bases:
            if isinstance(base, cls):
                raise TypeError("Cannot subclass %s" %
                                (', '.join(map(_type_repr, bases)) or '()'))

    def __init__(self, *args, **kwds):
        pass

    def _eval_type(self, globalns, localns):
        """Override this in subclasses to interpret forward references.

        For example, List['C'] is internally stored as
        List[_ForwardRef('C')], which should evaluate to List[C],
        where C is an object found in globalns or localns (searching
        localns first, of course).
        """
        return self

    def _get_type_vars(self, tvars):
        pass

    def __repr__(self):
        qname = _trim_name(_qualname(self))
        return '%s.%s' % (self.__module__, qname)


class _TypingBase(object):
    """Internal indicator of special typing constructs."""
    __metaclass__ = TypingMeta
    __slots__ = ('__weakref__',)

    def __init__(self, *args, **kwds):
        pass

    def __new__(cls, *args, **kwds):
        """Constructor.

        This only exists to give a better error message in case
        someone tries to subclass a special typing object (not a good idea).
        """
        if (len(args) == 3 and
                isinstance(args[0], str) and
                isinstance(args[1], tuple)):
            # Close enough.
            raise TypeError("Cannot subclass %r" % cls)
        return super(_TypingBase, cls).__new__(cls)

    # Things that are not classes also need these.
    def _eval_type(self, globalns, localns):
        return self

    def _get_type_vars(self, tvars):
        pass

    def __repr__(self):
        cls = type(self)
        qname = _trim_name(_qualname(cls))
        return '%s.%s' % (cls.__module__, qname)

    def __call__(self, *args, **kwds):
        raise TypeError("Cannot instantiate %r" % type(self))


class _FinalTypingBase(_TypingBase):
    """Internal mix-in class to prevent instantiation.

    Prevents instantiation unless _root=True is given in class call.
    It is used to create pseudo-singleton instances Any, Union, Optional, etc.
    """

    __slots__ = ()

    def __new__(cls, *args, **kwds):
        self = super(_FinalTypingBase, cls).__new__(cls, *args, **kwds)
        if '_root' in kwds and kwds['_root'] is True:
            return self
        raise TypeError("Cannot instantiate %r" % cls)

    def __reduce__(self):
        return _trim_name(type(self).__name__)


class _ForwardRef(_TypingBase):
    """Internal wrapper to hold a forward reference."""

    __slots__ = ('__forward_arg__', '__forward_code__',
                 '__forward_evaluated__', '__forward_value__')

    def __init__(self, arg):
        super(_ForwardRef, self).__init__(arg)
        if not isinstance(arg, basestring):
            raise TypeError('Forward reference must be a string -- got %r' % (arg,))
        try:
            code = compile(arg, '<string>', 'eval')
        except SyntaxError:
            raise SyntaxError('Forward reference must be an expression -- got %r' %
                              (arg,))
        self.__forward_arg__ = arg
        self.__forward_code__ = code
        self.__forward_evaluated__ = False
        self.__forward_value__ = None

    def _eval_type(self, globalns, localns):
        if not self.__forward_evaluated__ or localns is not globalns:
            if globalns is None and localns is None:
                globalns = localns = {}
            elif globalns is None:
                globalns = localns
            elif localns is None:
                localns = globalns
            self.__forward_value__ = _type_check(
                eval(self.__forward_code__, globalns, localns),
                "Forward references must evaluate to types.")
            self.__forward_evaluated__ = True
        return self.__forward_value__

    def __eq__(self, other):
        if not isinstance(other, _ForwardRef):
            return NotImplemented
        return (self.__forward_arg__ == other.__forward_arg__ and
                self.__forward_value__ == other.__forward_value__)

    def __hash__(self):
        return hash((self.__forward_arg__, self.__forward_value__))

    def __instancecheck__(self, obj):
        raise TypeError("Forward references cannot be used with isinstance().")

    def __subclasscheck__(self, cls):
        raise TypeError("Forward references cannot be used with issubclass().")

    def __repr__(self):
        return '_ForwardRef(%r)' % (self.__forward_arg__,)


class _TypeAlias(_TypingBase):
    """Internal helper class for defining generic variants of concrete types.

    Note that this is not a type; let's call it a pseudo-type.  It cannot
    be used in instance and subclass checks in parameterized form, i.e.
    ``isinstance(42, Match[str])`` raises ``TypeError`` instead of returning
    ``False``.
    """

    __slots__ = ('name', 'type_var', 'impl_type', 'type_checker')

    def __init__(self, name, type_var, impl_type, type_checker):
        """Initializer.

        Args:
            name: The name, e.g. 'Pattern'.
            type_var: The type parameter, e.g. AnyStr, or the
                specific type, e.g. str.
            impl_type: The implementation type.
            type_checker: Function that takes an impl_type instance.
                and returns a value that should be a type_var instance.
        """
        assert isinstance(name, basestring), repr(name)
        assert isinstance(impl_type, type), repr(impl_type)
        assert not isinstance(impl_type, TypingMeta), repr(impl_type)
        assert isinstance(type_var, (type, _TypingBase)), repr(type_var)
        self.name = name
        self.type_var = type_var
        self.impl_type = impl_type
        self.type_checker = type_checker

    def __repr__(self):
        return "%s[%s]" % (self.name, _type_repr(self.type_var))

    def __getitem__(self, parameter):
        if not isinstance(self.type_var, TypeVar):
            raise TypeError("%s cannot be further parameterized." % self)
        if self.type_var.__constraints__ and isinstance(parameter, type):
            if not issubclass(parameter, self.type_var.__constraints__):
                raise TypeError("%s is not a valid substitution for %s." %
                                (parameter, self.type_var))
        if isinstance(parameter, TypeVar) and parameter is not self.type_var:
            raise TypeError("%s cannot be re-parameterized." % self)
        return self.__class__(self.name, parameter,
                              self.impl_type, self.type_checker)

    def __eq__(self, other):
        if not isinstance(other, _TypeAlias):
            return NotImplemented
        return self.name == other.name and self.type_var == other.type_var

    def __hash__(self):
        return hash((self.name, self.type_var))

    def __instancecheck__(self, obj):
        if not isinstance(self.type_var, TypeVar):
            raise TypeError("Parameterized type aliases cannot be used "
                            "with isinstance().")
        return isinstance(obj, self.impl_type)

    def __subclasscheck__(self, cls):
        if not isinstance(self.type_var, TypeVar):
            raise TypeError("Parameterized type aliases cannot be used "
                            "with issubclass().")
        return issubclass(cls, self.impl_type)


def _get_type_vars(types, tvars):
    for t in types:
        if isinstance(t, TypingMeta) or isinstance(t, _TypingBase):
            t._get_type_vars(tvars)


def _type_vars(types):
    tvars = []
    _get_type_vars(types, tvars)
    return tuple(tvars)


def _eval_type(t, globalns, localns):
    if isinstance(t, TypingMeta) or isinstance(t, _TypingBase):
        return t._eval_type(globalns, localns)
    return t


def _type_check(arg, msg):
    """Check that the argument is a type, and return it (internal helper).

    As a special case, accept None and return type(None) instead.
    Also, _TypeAlias instances (e.g. Match, Pattern) are acceptable.

    The msg argument is a human-readable error message, e.g.

        "Union[arg, ...]: arg should be a type."

    We append the repr() of the actual value (truncated to 100 chars).
    """
    if arg is None:
        return type(None)
    if isinstance(arg, basestring):
        arg = _ForwardRef(arg)
    if (
        isinstance(arg, _TypingBase) and type(arg).__name__ == '_ClassVar' or
        not isinstance(arg, (type, _TypingBase)) and not callable(arg)
    ):
        raise TypeError(msg + " Got %.100r." % (arg,))
    # Bare Union etc. are not valid as type arguments
    if (
        type(arg).__name__ in ('_Union', '_Optional') and
        not getattr(arg, '__origin__', None) or
        isinstance(arg, TypingMeta) and arg._gorg in (Generic, Protocol)
    ):
        raise TypeError("Plain %s is not valid as type argument" % arg)
    return arg


def _type_repr(obj):
    """Return the repr() of an object, special-casing types (internal helper).

    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    if isinstance(obj, type) and not isinstance(obj, TypingMeta):
        if obj.__module__ == '__builtin__':
            return _qualname(obj)
        return '%s.%s' % (obj.__module__, _qualname(obj))
    if obj is Ellipsis:
        return '...'
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)


class ClassVarMeta(TypingMeta):
    """Metaclass for _ClassVar"""

    def __new__(cls, name, bases, namespace):
        cls.assert_no_subclassing(bases)
        self = super(ClassVarMeta, cls).__new__(cls, name, bases, namespace)
        return self


class _ClassVar(_FinalTypingBase):
    """Special type construct to mark class variables.

    An annotation wrapped in ClassVar indicates that a given
    attribute is intended to be used as a class variable and
    should not be set on instances of that class. Usage::

      class Starship:
          stats = {}  # type: ClassVar[Dict[str, int]] # class variable
          damage = 10 # type: int                      # instance variable

    ClassVar accepts only types and cannot be further subscribed.

    Note that ClassVar is not a class itself, and should not
    be used with isinstance() or issubclass().
    """

    __metaclass__ = ClassVarMeta
    __slots__ = ('__type__',)

    def __init__(self, tp=None, _root=False):
        self.__type__ = tp

    def __getitem__(self, item):
        cls = type(self)
        if self.__type__ is None:
            return cls(_type_check(item,
                       '{} accepts only types.'.format(cls.__name__[1:])),
                       _root=True)
        raise TypeError('{} cannot be further subscripted'
                        .format(cls.__name__[1:]))

    def _eval_type(self, globalns, localns):
        return type(self)(_eval_type(self.__type__, globalns, localns),
                          _root=True)

    def __repr__(self):
        r = super(_ClassVar, self).__repr__()
        if self.__type__ is not None:
            r += '[{}]'.format(_type_repr(self.__type__))
        return r

    def __hash__(self):
        return hash((type(self).__name__, self.__type__))

    def __eq__(self, other):
        if not isinstance(other, _ClassVar):
            return NotImplemented
        if self.__type__ is not None:
            return self.__type__ == other.__type__
        return self is other


ClassVar = _ClassVar(_root=True)


class _FinalMeta(TypingMeta):
    """Metaclass for _Final"""

    def __new__(cls, name, bases, namespace):
        cls.assert_no_subclassing(bases)
        self = super(_FinalMeta, cls).__new__(cls, name, bases, namespace)
        return self


class _Final(_FinalTypingBase):
    """A special typing construct to indicate that a name
    cannot be re-assigned or overridden in a subclass.
    For example:

        MAX_SIZE: Final = 9000
        MAX_SIZE += 1  # Error reported by type checker

        class Connection:
            TIMEOUT: Final[int] = 10
        class FastConnector(Connection):
            TIMEOUT = 1  # Error reported by type checker

    There is no runtime checking of these properties.
    """

    __metaclass__ = _FinalMeta
    __slots__ = ('__type__',)

    def __init__(self, tp=None, **kwds):
        self.__type__ = tp

    def __getitem__(self, item):
        cls = type(self)
        if self.__type__ is None:
            return cls(_type_check(item,
                       '{} accepts only single type.'.format(cls.__name__[1:])),
                       _root=True)
        raise TypeError('{} cannot be further subscripted'
                        .format(cls.__name__[1:]))

    def _eval_type(self, globalns, localns):
        new_tp = _eval_type(self.__type__, globalns, localns)
        if new_tp == self.__type__:
            return self
        return type(self)(new_tp, _root=True)

    def __repr__(self):
        r = super(_Final, self).__repr__()
        if self.__type__ is not None:
            r += '[{}]'.format(_type_repr(self.__type__))
        return r

    def __hash__(self):
        return hash((type(self).__name__, self.__type__))

    def __eq__(self, other):
        if not isinstance(other, _Final):
            return NotImplemented
        if self.__type__ is not None:
            return self.__type__ == other.__type__
        return self is other


Final = _Final(_root=True)


def final(f):
    """This decorator can be used to indicate to type checkers that
    the decorated method cannot be overridden, and decorated class
    cannot be subclassed. For example:

        class Base:
            @final
            def done(self) -> None:
                ...
        class Sub(Base):
            def done(self) -> None:  # Error reported by type checker
                ...
        @final
        class Leaf:
            ...
        class Other(Leaf):  # Error reported by type checker
            ...

    There is no runtime checking of these properties.
    """
    return f


class _LiteralMeta(TypingMeta):
    """Metaclass for _Literal"""

    def __new__(cls, name, bases, namespace):
        cls.assert_no_subclassing(bases)
        self = super(_LiteralMeta, cls).__new__(cls, name, bases, namespace)
        return self


class _Literal(_FinalTypingBase):
    """A type that can be used to indicate to type checkers that the
    corresponding value has a value literally equivalent to the
    provided parameter. For example:

        var: Literal[4] = 4

    The type checker understands that 'var' is literally equal to the
    value 4 and no other value.

    Literal[...] cannot be subclassed. There is no runtime checking
    verifying that the parameter is actually a value instead of a type.
    """

    __metaclass__ = _LiteralMeta
    __slots__ = ('__values__',)

    def __init__(self, values=None, **kwds):
        self.__values__ = values

    def __getitem__(self, item):
        cls = type(self)
        if self.__values__ is None:
            if not isinstance(item, tuple):
                item = (item,)
            return cls(values=item,
                       _root=True)
        raise TypeError('{} cannot be further subscripted'
                        .format(cls.__name__[1:]))

    def _eval_type(self, globalns, localns):
        return self

    def __repr__(self):
        r = super(_Literal, self).__repr__()
        if self.__values__ is not None:
            r += '[{}]'.format(', '.join(map(_type_repr, self.__values__)))
        return r

    def __hash__(self):
        return hash((type(self).__name__, self.__values__))

    def __eq__(self, other):
        if not isinstance(other, _Literal):
            return NotImplemented
        if self.__values__ is not None:
            return self.__values__ == other.__values__
        return self is other


Literal = _Literal(_root=True)


class AnyMeta(TypingMeta):
    """Metaclass for Any."""

    def __new__(cls, name, bases, namespace):
        cls.assert_no_subclassing(bases)
        self = super(AnyMeta, cls).__new__(cls, name, bases, namespace)
        return self


class _Any(_FinalTypingBase):
    """Special type indicating an unconstrained type.

    - Any is compatible with every type.
    - Any assumed to have all methods.
    - All values assumed to be instances of Any.

    Note that all the above statements are true from the point of view of
    static type checkers. At runtime, Any should not be used with instance
    or class checks.
    """
    __metaclass__ = AnyMeta
    __slots__ = ()

    def __instancecheck__(self, obj):
        raise TypeError("Any cannot be used with isinstance().")

    def __subclasscheck__(self, cls):
        raise TypeError("Any cannot be used with issubclass().")


Any = _Any(_root=True)


class NoReturnMeta(TypingMeta):
    """Metaclass for NoReturn."""

    def __new__(cls, name, bases, namespace):
        cls.assert_no_subclassing(bases)
        self = super(NoReturnMeta, cls).__new__(cls, name, bases, namespace)
        return self


class _NoReturn(_FinalTypingBase):
    """Special type indicating functions that never return.
    Example::

      from typing import NoReturn

      def stop() -> NoReturn:
          raise Exception('no way')

    This type is invalid in other positions, e.g., ``List[NoReturn]``
    will fail in static type checkers.
    """
    __metaclass__ = NoReturnMeta
    __slots__ = ()

    def __instancecheck__(self, obj):
        raise TypeError("NoReturn cannot be used with isinstance().")

    def __subclasscheck__(self, cls):
        raise TypeError("NoReturn cannot be used with issubclass().")


NoReturn = _NoReturn(_root=True)


class TypeVarMeta(TypingMeta):
    def __new__(cls, name, bases, namespace):
        cls.assert_no_subclassing(bases)
        return super(TypeVarMeta, cls).__new__(cls, name, bases, namespace)


class TypeVar(_TypingBase):
    """Type variable.

    Usage::

      T = TypeVar('T')  # Can be anything
      A = TypeVar('A', str, bytes)  # Must be str or bytes

    Type variables exist primarily for the benefit of static type
    checkers.  They serve as the parameters for generic types as well
    as for generic function definitions.  See class Generic for more
    information on generic types.  Generic functions work as follows:

      def repeat(x: T, n: int) -> List[T]:
          '''Return a list containing n references to x.'''
          return [x]*n

      def longest(x: A, y: A) -> A:
          '''Return the longest of two strings.'''
          return x if len(x) >= len(y) else y

    The latter example's signature is essentially the overloading
    of (str, str) -> str and (bytes, bytes) -> bytes.  Also note
    that if the arguments are instances of some subclass of str,
    the return type is still plain str.

    At runtime, isinstance(x, T) and issubclass(C, T) will raise TypeError.

    Type variables defined with covariant=True or contravariant=True
    can be used do declare covariant or contravariant generic types.
    See PEP 484 for more details. By default generic types are invariant
    in all type variables.

    Type variables can be introspected. e.g.:

      T.__name__ == 'T'
      T.__constraints__ == ()
      T.__covariant__ == False
      T.__contravariant__ = False
      A.__constraints__ == (str, bytes)
    """

    __metaclass__ = TypeVarMeta
    __slots__ = ('__name__', '__bound__', '__constraints__',
                 '__covariant__', '__contravariant__')

    def __init__(self, name, *constraints, **kwargs):
        super(TypeVar, self).__init__(name, *constraints, **kwargs)
        bound = kwargs.get('bound', None)
        covariant = kwargs.get('covariant', False)
        contravariant = kwargs.get('contravariant', False)
        self.__name__ = name
        if covariant and contravariant:
            raise ValueError("Bivariant types are not supported.")
        self.__covariant__ = bool(covariant)
        self.__contravariant__ = bool(contravariant)
        if constraints and bound is not None:
            raise TypeError("Constraints cannot be combined with bound=...")
        if constraints and len(constraints) == 1:
            raise TypeError("A single constraint is not allowed")
        msg = "TypeVar(name, constraint, ...): constraints must be types."
        self.__constraints__ = tuple(_type_check(t, msg) for t in constraints)
        if bound:
            self.__bound__ = _type_check(bound, "Bound must be a type.")
        else:
            self.__bound__ = None

    def _get_type_vars(self, tvars):
        if self not in tvars:
            tvars.append(self)

    def __repr__(self):
        if self.__covariant__:
            prefix = '+'
        elif self.__contravariant__:
            prefix = '-'
        else:
            prefix = '~'
        return prefix + self.__name__

    def __instancecheck__(self, instance):
        raise TypeError("Type variables cannot be used with isinstance().")

    def __subclasscheck__(self, cls):
        raise TypeError("Type variables cannot be used with issubclass().")


# Some unconstrained type variables.  These are used by the container types.
# (These are not for export.)
T = TypeVar('T')  # Any type.
KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.
T_co = TypeVar('T_co', covariant=True)  # Any type covariant containers.
V_co = TypeVar('V_co', covariant=True)  # Any type covariant containers.
VT_co = TypeVar('VT_co', covariant=True)  # Value type covariant containers.
T_contra = TypeVar('T_contra', contravariant=True)  # Ditto contravariant.

# A useful type variable with constraints.  This represents string types.
# (This one *is* for export!)
AnyStr = TypeVar('AnyStr', bytes, unicode)


def _replace_arg(arg, tvars, args):
    """An internal helper function: replace arg if it is a type variable
    found in tvars with corresponding substitution from args or
    with corresponding substitution sub-tree if arg is a generic type.
    """

    if tvars is None:
        tvars = []
    if hasattr(arg, '_subs_tree') and isinstance(arg, (GenericMeta, _TypingBase)):
        return arg._subs_tree(tvars, args)
    if isinstance(arg, TypeVar):
        for i, tvar in enumerate(tvars):
            if arg == tvar:
                return args[i]
    return arg


# Special typing constructs Union, Optional, Generic, Callable and Tuple
# use three special attributes for internal bookkeeping of generic types:
# * __parameters__ is a tuple of unique free type parameters of a generic
#   type, for example, Dict[T, T].__parameters__ == (T,);
# * __origin__ keeps a reference to a type that was subscripted,
#   e.g., Union[T, int].__origin__ == Union;
# * __args__ is a tuple of all arguments used in subscripting,
#   e.g., Dict[T, int].__args__ == (T, int).


def _subs_tree(cls, tvars=None, args=None):
    """An internal helper function: calculate substitution tree
    for generic cls after replacing its type parameters with
    substitutions in tvars -> args (if any).
    Repeat the same following __origin__'s.

    Return a list of arguments with all possible substitutions
    performed. Arguments that are generic classes themselves are represented
    as tuples (so that no new classes are created by this function).
    For example: _subs_tree(List[Tuple[int, T]][str]) == [(Tuple, int, str)]
    """

    if cls.__origin__ is None:
        return cls
    # Make of chain of origins (i.e. cls -> cls.__origin__)
    current = cls.__origin__
    orig_chain = []
    while current.__origin__ is not None:
        orig_chain.append(current)
        current = current.__origin__
    # Replace type variables in __args__ if asked ...
    tree_args = []
    for arg in cls.__args__:
        tree_args.append(_replace_arg(arg, tvars, args))
    # ... then continue replacing down the origin chain.
    for ocls in orig_chain:
        new_tree_args = []
        for arg in ocls.__args__:
            new_tree_args.append(_replace_arg(arg, ocls.__parameters__, tree_args))
        tree_args = new_tree_args
    return tree_args


def _remove_dups_flatten(parameters):
    """An internal helper for Union creation and substitution: flatten Union's
    among parameters, then remove duplicates and strict subclasses.
    """

    # Flatten out Union[Union[...], ...].
    params = []
    for p in parameters:
        if isinstance(p, _Union) and p.__origin__ is Union:
            params.extend(p.__args__)
        elif isinstance(p, tuple) and len(p) > 0 and p[0] is Union:
            params.extend(p[1:])
        else:
            params.append(p)
    # Weed out strict duplicates, preserving the first of each occurrence.
    all_params = set(params)
    if len(all_params) < len(params):
        new_params = []
        for t in params:
            if t in all_params:
                new_params.append(t)
                all_params.remove(t)
        params = new_params
        assert not all_params, all_params
    # Weed out subclasses.
    # E.g. Union[int, Employee, Manager] == Union[int, Employee].
    # If object is present it will be sole survivor among proper classes.
    # Never discard type variables.
    # (In particular, Union[str, AnyStr] != AnyStr.)
    all_params = set(params)
    for t1 in params:
        if not isinstance(t1, type):
            continue
        if any(isinstance(t2, type) and issubclass(t1, t2)
               for t2 in all_params - {t1}
               if not (isinstance(t2, GenericMeta) and
                       t2.__origin__ is not None)):
            all_params.remove(t1)
    return tuple(t for t in params if t in all_params)


def _check_generic(cls, parameters):
    # Check correct count for parameters of a generic cls (internal helper).
    if not cls.__parameters__:
        raise TypeError("%s is not a generic class" % repr(cls))
    alen = len(parameters)
    elen = len(cls.__parameters__)
    if alen != elen:
        raise TypeError("Too %s parameters for %s; actual %s, expected %s" %
                        ("many" if alen > elen else "few", repr(cls), alen, elen))


_cleanups = []


def _tp_cache(func):
    maxsize = 128
    cache = {}
    _cleanups.append(cache.clear)

    @functools.wraps(func)
    def inner(*args):
        key = args
        try:
            return cache[key]
        except TypeError:
            # Assume it's an unhashable argument.
            return func(*args)
        except KeyError:
            value = func(*args)
            if len(cache) >= maxsize:
                # If the cache grows too much, just start over.
                cache.clear()
            cache[key] = value
            return value

    return inner


class UnionMeta(TypingMeta):
    """Metaclass for Union."""

    def __new__(cls, name, bases, namespace):
        cls.assert_no_subclassing(bases)
        return super(UnionMeta, cls).__new__(cls, name, bases, namespace)


class _Union(_FinalTypingBase):
    """Union type; Union[X, Y] means either X or Y.

    To define a union, use e.g. Union[int, str].  Details:

    - The arguments must be types and there must be at least one.

    - None as an argument is a special case and is replaced by
      type(None).

    - Unions of unions are flattened, e.g.::

        Union[Union[int, str], float] == Union[int, str, float]

    - Unions of a single argument vanish, e.g.::

        Union[int] == int  # The constructor actually returns int

    - Redundant arguments are skipped, e.g.::

        Union[int, str, int] == Union[int, str]

    - When comparing unions, the argument order is ignored, e.g.::

        Union[int, str] == Union[str, int]

    - When two arguments have a subclass relationship, the least
      derived argument is kept, e.g.::

        class Employee: pass
        class Manager(Employee): pass
        Union[int, Employee, Manager] == Union[int, Employee]
        Union[Manager, int, Employee] == Union[int, Employee]
        Union[Employee, Manager] == Employee

    - Similar for object::

        Union[int, object] == object

    - You cannot subclass or instantiate a union.

    - You can use Optional[X] as a shorthand for Union[X, None].
    """

    __metaclass__ = UnionMeta
    __slots__ = ('__parameters__', '__args__', '__origin__', '__tree_hash__')

    def __new__(cls, parameters=None, origin=None, *args, **kwds):
        self = super(_Union, cls).__new__(cls, parameters, origin, *args, **kwds)
        if origin is None:
            self.__parameters__ = None
            self.__args__ = None
            self.__origin__ = None
            self.__tree_hash__ = hash(frozenset(('Union',)))
            return self
        if not isinstance(parameters, tuple):
            raise TypeError("Expected parameters=<tuple>")
        if origin is Union:
            parameters = _remove_dups_flatten(parameters)
            # It's not a union if there's only one type left.
            if len(parameters) == 1:
                return parameters[0]
        self.__parameters__ = _type_vars(parameters)
        self.__args__ = parameters
        self.__origin__ = origin
        # Pre-calculate the __hash__ on instantiation.
        # This improves speed for complex substitutions.
        subs_tree = self._subs_tree()
        if isinstance(subs_tree, tuple):
            self.__tree_hash__ = hash(frozenset(subs_tree))
        else:
            self.__tree_hash__ = hash(subs_tree)
        return self

    def _eval_type(self, globalns, localns):
        if self.__args__ is None:
            return self
        ev_args = tuple(_eval_type(t, globalns, localns) for t in self.__args__)
        ev_origin = _eval_type(self.__origin__, globalns, localns)
        if ev_args == self.__args__ and ev_origin == self.__origin__:
            # Everything is already evaluated.
            return self
        return self.__class__(ev_args, ev_origin, _root=True)

    def _get_type_vars(self, tvars):
        if self.__origin__ and self.__parameters__:
            _get_type_vars(self.__parameters__, tvars)

    def __repr__(self):
        if self.__origin__ is None:
            return super(_Union, self).__repr__()
        tree = self._subs_tree()
        if not isinstance(tree, tuple):
            return repr(tree)
        return tree[0]._tree_repr(tree)

    def _tree_repr(self, tree):
        arg_list = []
        for arg in tree[1:]:
            if not isinstance(arg, tuple):
                arg_list.append(_type_repr(arg))
            else:
                arg_list.append(arg[0]._tree_repr(arg))
        return super(_Union, self).__repr__() + '[%s]' % ', '.join(arg_list)

    @_tp_cache
    def __getitem__(self, parameters):
        if parameters == ():
            raise TypeError("Cannot take a Union of no types.")
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        if self.__origin__ is None:
            msg = "Union[arg, ...]: each arg must be a type."
        else:
            msg = "Parameters to generic types must be types."
        parameters = tuple(_type_check(p, msg) for p in parameters)
        if self is not Union:
            _check_generic(self, parameters)
        return self.__class__(parameters, origin=self, _root=True)

    def _subs_tree(self, tvars=None, args=None):
        if self is Union:
            return Union  # Nothing to substitute
        tree_args = _subs_tree(self, tvars, args)
        tree_args = _remove_dups_flatten(tree_args)
        if len(tree_args) == 1:
            return tree_args[0]  # Union of a single type is that type
        return (Union,) + tree_args

    def __eq__(self, other):
        if isinstance(other, _Union):
            return self.__tree_hash__ == other.__tree_hash__
        elif self is not Union:
            return self._subs_tree() == other
        else:
            return self is other

    def __hash__(self):
        return self.__tree_hash__

    def __instancecheck__(self, obj):
        raise TypeError("Unions cannot be used with isinstance().")

    def __subclasscheck__(self, cls):
        raise TypeError("Unions cannot be used with issubclass().")


Union = _Union(_root=True)


class OptionalMeta(TypingMeta):
    """Metaclass for Optional."""

    def __new__(cls, name, bases, namespace):
        cls.assert_no_subclassing(bases)
        return super(OptionalMeta, cls).__new__(cls, name, bases, namespace)


class _Optional(_FinalTypingBase):
    """Optional type.

    Optional[X] is equivalent to Union[X, None].
    """

    __metaclass__ = OptionalMeta
    __slots__ = ()

    @_tp_cache
    def __getitem__(self, arg):
        arg = _type_check(arg, "Optional[t] requires a single type.")
        return Union[arg, type(None)]


Optional = _Optional(_root=True)


def _next_in_mro(cls):
    """Helper for Generic.__new__.

    Returns the class after the last occurrence of Generic or
    Generic[...] in cls.__mro__.
    """
    next_in_mro = object
    # Look for the last occurrence of Generic or Generic[...].
    for i, c in enumerate(cls.__mro__[:-1]):
        if isinstance(c, GenericMeta) and c._gorg is Generic:
            next_in_mro = cls.__mro__[i + 1]
    return next_in_mro


def _make_subclasshook(cls):
    """Construct a __subclasshook__ callable that incorporates
    the associated __extra__ class in subclass checks performed
    against cls.
    """
    if isinstance(cls.__extra__, abc.ABCMeta):
        # The logic mirrors that of ABCMeta.__subclasscheck__.
        # Registered classes need not be checked here because
        # cls and its extra share the same _abc_registry.
        def __extrahook__(cls, subclass):
            res = cls.__extra__.__subclasshook__(subclass)
            if res is not NotImplemented:
                return res
            if cls.__extra__ in getattr(subclass, '__mro__', ()):
                return True
            for scls in cls.__extra__.__subclasses__():
                if isinstance(scls, GenericMeta):
                    continue
                if issubclass(subclass, scls):
                    return True
            return NotImplemented
    else:
        # For non-ABC extras we'll just call issubclass().
        def __extrahook__(cls, subclass):
            if cls.__extra__ and issubclass(subclass, cls.__extra__):
                return True
            return NotImplemented
    return classmethod(__extrahook__)


class GenericMeta(TypingMeta, abc.ABCMeta):
    """Metaclass for generic types.

    This is a metaclass for typing.Generic and generic ABCs defined in
    typing module. User defined subclasses of GenericMeta can override
    __new__ and invoke super().__new__. Note that GenericMeta.__new__
    has strict rules on what is allowed in its bases argument:
    * plain Generic is disallowed in bases;
    * Generic[...] should appear in bases at most once;
    * if Generic[...] is present, then it should list all type variables
      that appear in other bases.
    In addition, type of all generic bases is erased, e.g., C[int] is
    stripped to plain C.
    """

    def __new__(cls, name, bases, namespace,
                tvars=None, args=None, origin=None, extra=None, orig_bases=None):
        """Create a new generic class. GenericMeta.__new__ accepts
        keyword arguments that are used for internal bookkeeping, therefore
        an override should pass unused keyword arguments to super().
        """
        if tvars is not None:
            # Called from __getitem__() below.
            assert origin is not None
            assert all(isinstance(t, TypeVar) for t in tvars), tvars
        else:
            # Called from class statement.
            assert tvars is None, tvars
            assert args is None, args
            assert origin is None, origin

            # Get the full set of tvars from the bases.
            tvars = _type_vars(bases)
            # Look for Generic[T1, ..., Tn].
            # If found, tvars must be a subset of it.
            # If not found, tvars is it.
            # Also check for and reject plain Generic,
            # and reject multiple Generic[...].
            gvars = None
            for base in bases:
                if base is Generic:
                    raise TypeError("Cannot inherit from plain Generic")
                if (isinstance(base, GenericMeta) and
                        base.__origin__ in (Generic, Protocol)):
                    if gvars is not None:
                        raise TypeError(
                            "Cannot inherit from Generic[...] or"
                            " Protocol[...] multiple times.")
                    gvars = base.__parameters__
            if gvars is None:
                gvars = tvars
            else:
                tvarset = set(tvars)
                gvarset = set(gvars)
                if not tvarset <= gvarset:
                    raise TypeError(
                        "Some type variables (%s) "
                        "are not listed in %s[%s]" %
                        (", ".join(str(t) for t in tvars if t not in gvarset),
                         "Generic" if any(b.__origin__ is Generic
                                          for b in bases) else "Protocol",
                         ", ".join(str(g) for g in gvars)))
                tvars = gvars

        initial_bases = bases
        if extra is None:
            extra = namespace.get('__extra__')
        if extra is not None and type(extra) is abc.ABCMeta and extra not in bases:
            bases = (extra,) + bases
        bases = tuple(b._gorg if isinstance(b, GenericMeta) else b for b in bases)

        # remove bare Generic from bases if there are other generic bases
        if any(isinstance(b, GenericMeta) and b is not Generic for b in bases):
            bases = tuple(b for b in bases if b is not Generic)
        namespace.update({'__origin__': origin, '__extra__': extra})
        self = super(GenericMeta, cls).__new__(cls, name, bases, namespace)
        super(GenericMeta, self).__setattr__('_gorg',
                                             self if not origin else origin._gorg)

        self.__parameters__ = tvars
        # Be prepared that GenericMeta will be subclassed by TupleMeta
        # and CallableMeta, those two allow ..., (), or [] in __args___.
        self.__args__ = tuple(Ellipsis if a is _TypingEllipsis else
                              () if a is _TypingEmpty else
                              a for a in args) if args else None
        # Speed hack (https://github.com/python/typing/issues/196).
        self.__next_in_mro__ = _next_in_mro(self)
        # Preserve base classes on subclassing (__bases__ are type erased now).
        if orig_bases is None:
            self.__orig_bases__ = initial_bases

        # This allows unparameterized generic collections to be used
        # with issubclass() and isinstance() in the same way as their
        # collections.abc counterparts (e.g., isinstance([], Iterable)).
        if (
            '__subclasshook__' not in namespace and extra or
            # allow overriding
            getattr(self.__subclasshook__, '__name__', '') == '__extrahook__'
        ):
            self.__subclasshook__ = _make_subclasshook(self)

        if origin and hasattr(origin, '__qualname__'):  # Fix for Python 3.2.
            self.__qualname__ = origin.__qualname__
        self.__tree_hash__ = (hash(self._subs_tree()) if origin else
                              super(GenericMeta, self).__hash__())
        return self

    def __init__(self, *args, **kwargs):
        super(GenericMeta, self).__init__(*args, **kwargs)
        if isinstance(self.__extra__, abc.ABCMeta):
            self._abc_registry = self.__extra__._abc_registry
            self._abc_cache = self.__extra__._abc_cache
        elif self.__origin__ is not None:
            self._abc_registry = self.__origin__._abc_registry
            self._abc_cache = self.__origin__._abc_cache

    # _abc_negative_cache and _abc_negative_cache_version
    # realised as descriptors, since GenClass[t1, t2, ...] always
    # share subclass info with GenClass.
    # This is an important memory optimization.
    @property
    def _abc_negative_cache(self):
        if isinstance(self.__extra__, abc.ABCMeta):
            return self.__extra__._abc_negative_cache
        return self._gorg._abc_generic_negative_cache

    @_abc_negative_cache.setter
    def _abc_negative_cache(self, value):
        if self.__origin__ is None:
            if isinstance(self.__extra__, abc.ABCMeta):
                self.__extra__._abc_negative_cache = value
            else:
                self._abc_generic_negative_cache = value

    @property
    def _abc_negative_cache_version(self):
        if isinstance(self.__extra__, abc.ABCMeta):
            return self.__extra__._abc_negative_cache_version
        return self._gorg._abc_generic_negative_cache_version

    @_abc_negative_cache_version.setter
    def _abc_negative_cache_version(self, value):
        if self.__origin__ is None:
            if isinstance(self.__extra__, abc.ABCMeta):
                self.__extra__._abc_negative_cache_version = value
            else:
                self._abc_generic_negative_cache_version = value

    def _get_type_vars(self, tvars):
        if self.__origin__ and self.__parameters__:
            _get_type_vars(self.__parameters__, tvars)

    def _eval_type(self, globalns, localns):
        ev_origin = (self.__origin__._eval_type(globalns, localns)
                     if self.__origin__ else None)
        ev_args = tuple(_eval_type(a, globalns, localns) for a
                        in self.__args__) if self.__args__ else None
        if ev_origin == self.__origin__ and ev_args == self.__args__:
            return self
        return self.__class__(self.__name__,
                              self.__bases__,
                              dict(self.__dict__),
                              tvars=_type_vars(ev_args) if ev_args else None,
                              args=ev_args,
                              origin=ev_origin,
                              extra=self.__extra__,
                              orig_bases=self.__orig_bases__)

    def __repr__(self):
        if self.__origin__ is None:
            return super(GenericMeta, self).__repr__()
        return self._tree_repr(self._subs_tree())

    def _tree_repr(self, tree):
        arg_list = []
        for arg in tree[1:]:
            if arg == ():
                arg_list.append('()')
            elif not isinstance(arg, tuple):
                arg_list.append(_type_repr(arg))
            else:
                arg_list.append(arg[0]._tree_repr(arg))
        return super(GenericMeta, self).__repr__() + '[%s]' % ', '.join(arg_list)

    def _subs_tree(self, tvars=None, args=None):
        if self.__origin__ is None:
            return self
        tree_args = _subs_tree(self, tvars, args)
        return (self._gorg,) + tuple(tree_args)

    def __eq__(self, other):
        if not isinstance(other, GenericMeta):
            return NotImplemented
        if self.__origin__ is None or other.__origin__ is None:
            return self is other
        return self.__tree_hash__ == other.__tree_hash__

    def __hash__(self):
        return self.__tree_hash__

    @_tp_cache
    def __getitem__(self, params):
        if not isinstance(params, tuple):
            params = (params,)
        if not params and self._gorg is not Tuple:
            raise TypeError(
                "Parameter list to %s[...] cannot be empty" % _qualname(self))
        msg = "Parameters to generic types must be types."
        params = tuple(_type_check(p, msg) for p in params)
        if self in (Generic, Protocol):
            # Generic can only be subscripted with unique type variables.
            if not all(isinstance(p, TypeVar) for p in params):
                raise TypeError(
                    "Parameters to %s[...] must all be type variables" % self.__name__)
            if len(set(params)) != len(params):
                raise TypeError(
                    "Parameters to %s[...] must all be unique" % self.__name__)
            tvars = params
            args = params
        elif self in (Tuple, Callable):
            tvars = _type_vars(params)
            args = params
        elif self.__origin__ in (Generic, Protocol):
            # Can't subscript Generic[...] or Protocol[...].
            raise TypeError("Cannot subscript already-subscripted %s" %
                            repr(self))
        else:
            # Subscripting a regular Generic subclass.
            _check_generic(self, params)
            tvars = _type_vars(params)
            args = params

        prepend = (self,) if self.__origin__ is None else ()
        return self.__class__(self.__name__,
                              prepend + self.__bases__,
                              dict(self.__dict__),
                              tvars=tvars,
                              args=args,
                              origin=self,
                              extra=self.__extra__,
                              orig_bases=self.__orig_bases__)

    def __subclasscheck__(self, cls):
        if self.__origin__ is not None:
            # These should only be modules within the standard library.
            # singledispatch is an exception, because it's a Python 2 backport
            # of functools.singledispatch.
            whitelist = ['abc', 'functools', 'singledispatch']
            if (sys._getframe(1).f_globals['__name__'] in whitelist or
                    # The second frame is needed for the case where we came
                    # from _ProtocolMeta.__subclasscheck__.
                    sys._getframe(2).f_globals['__name__'] in whitelist):
                return False
            raise TypeError("Parameterized generics cannot be used with class "
                            "or instance checks")
        if self is Generic:
            raise TypeError("Class %r cannot be used with class "
                            "or instance checks" % self)
        return super(GenericMeta, self).__subclasscheck__(cls)

    def __instancecheck__(self, instance):
        # Since we extend ABC.__subclasscheck__ and
        # ABC.__instancecheck__ inlines the cache checking done by the
        # latter, we must extend __instancecheck__ too. For simplicity
        # we just skip the cache check -- instance checks for generic
        # classes are supposed to be rare anyways.
        if hasattr(instance, "__class__"):
            return issubclass(instance.__class__, self)
        return False

    def __setattr__(self, attr, value):
        # We consider all the subscripted genrics as proxies for original class
        if (
            attr.startswith('__') and attr.endswith('__') or
            attr.startswith('_abc_')
        ):
            super(GenericMeta, self).__setattr__(attr, value)
        else:
            super(GenericMeta, self._gorg).__setattr__(attr, value)


def _copy_generic(self):
    """Hack to work around https://bugs.python.org/issue11480 on Python 2"""
    return self.__class__(self.__name__, self.__bases__, dict(self.__dict__),
                          self.__parameters__, self.__args__, self.__origin__,
                          self.__extra__, self.__orig_bases__)


copy._copy_dispatch[GenericMeta] = _copy_generic


# Prevent checks for Generic to crash when defining Generic.
Generic = None


def _generic_new(base_cls, cls, *args, **kwds):
    # Assure type is erased on instantiation,
    # but attempt to store it in __orig_class__
    if cls.__origin__ is None:
        if (base_cls.__new__ is object.__new__ and
                cls.__init__ is not object.__init__):
            return base_cls.__new__(cls)
        else:
            return base_cls.__new__(cls, *args, **kwds)
    else:
        origin = cls._gorg
        if (base_cls.__new__ is object.__new__ and
                cls.__init__ is not object.__init__):
            obj = base_cls.__new__(origin)
        else:
            obj = base_cls.__new__(origin, *args, **kwds)
        try:
            obj.__orig_class__ = cls
        except AttributeError:
            pass
        obj.__init__(*args, **kwds)
        return obj


class Generic(object):
    """Abstract base class for generic types.

    A generic type is typically declared by inheriting from
    this class parameterized with one or more type variables.
    For example, a generic mapping type might be defined as::

      class Mapping(Generic[KT, VT]):
          def __getitem__(self, key: KT) -> VT:
              ...
          # Etc.

    This class can then be used as follows::

      def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
          try:
              return mapping[key]
          except KeyError:
              return default
    """

    __metaclass__ = GenericMeta
    __slots__ = ()

    def __new__(cls, *args, **kwds):
        if cls._gorg is Generic:
            raise TypeError("Type Generic cannot be instantiated; "
                            "it can be used only as a base class")
        return _generic_new(cls.__next_in_mro__, cls, *args, **kwds)


class _TypingEmpty(object):
    """Internal placeholder for () or []. Used by TupleMeta and CallableMeta
    to allow empty list/tuple in specific places, without allowing them
    to sneak in where prohibited.
    """


class _TypingEllipsis(object):
    """Internal placeholder for ... (ellipsis)."""


class TupleMeta(GenericMeta):
    """Metaclass for Tuple (internal)."""

    @_tp_cache
    def __getitem__(self, parameters):
        if self.__origin__ is not None or self._gorg is not Tuple:
            # Normal generic rules apply if this is not the first subscription
            # or a subscription of a subclass.
            return super(TupleMeta, self).__getitem__(parameters)
        if parameters == ():
            return super(TupleMeta, self).__getitem__((_TypingEmpty,))
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        if len(parameters) == 2 and parameters[1] is Ellipsis:
            msg = "Tuple[t, ...]: t must be a type."
            p = _type_check(parameters[0], msg)
            return super(TupleMeta, self).__getitem__((p, _TypingEllipsis))
        msg = "Tuple[t0, t1, ...]: each t must be a type."
        parameters = tuple(_type_check(p, msg) for p in parameters)
        return super(TupleMeta, self).__getitem__(parameters)

    def __instancecheck__(self, obj):
        if self.__args__ is None:
            return isinstance(obj, tuple)
        raise TypeError("Parameterized Tuple cannot be used "
                        "with isinstance().")

    def __subclasscheck__(self, cls):
        if self.__args__ is None:
            return issubclass(cls, tuple)
        raise TypeError("Parameterized Tuple cannot be used "
                        "with issubclass().")


copy._copy_dispatch[TupleMeta] = _copy_generic


class Tuple(tuple):
    """Tuple type; Tuple[X, Y] is the cross-product type of X and Y.

    Example: Tuple[T1, T2] is a tuple of two elements corresponding
    to type variables T1 and T2.  Tuple[int, float, str] is a tuple
    of an int, a float and a string.

    To specify a variable-length tuple of homogeneous type, use Tuple[T, ...].
    """

    __metaclass__ = TupleMeta
    __extra__ = tuple
    __slots__ = ()

    def __new__(cls, *args, **kwds):
        if cls._gorg is Tuple:
            raise TypeError("Type Tuple cannot be instantiated; "
                            "use tuple() instead")
        return _generic_new(tuple, cls, *args, **kwds)


class CallableMeta(GenericMeta):
    """ Metaclass for Callable."""

    def __repr__(self):
        if self.__origin__ is None:
            return super(CallableMeta, self).__repr__()
        return self._tree_repr(self._subs_tree())

    def _tree_repr(self, tree):
        if self._gorg is not Callable:
            return super(CallableMeta, self)._tree_repr(tree)
        # For actual Callable (not its subclass) we override
        # super(CallableMeta, self)._tree_repr() for nice formatting.
        arg_list = []
        for arg in tree[1:]:
            if not isinstance(arg, tuple):
                arg_list.append(_type_repr(arg))
            else:
                arg_list.append(arg[0]._tree_repr(arg))
        if arg_list[0] == '...':
            return repr(tree[0]) + '[..., %s]' % arg_list[1]
        return (repr(tree[0]) +
                '[[%s], %s]' % (', '.join(arg_list[:-1]), arg_list[-1]))

    def __getitem__(self, parameters):
        """A thin wrapper around __getitem_inner__ to provide the latter
        with hashable arguments to improve speed.
        """

        if self.__origin__ is not None or self._gorg is not Callable:
            return super(CallableMeta, self).__getitem__(parameters)
        if not isinstance(parameters, tuple) or len(parameters) != 2:
            raise TypeError("Callable must be used as "
                            "Callable[[arg, ...], result].")
        args, result = parameters
        if args is Ellipsis:
            parameters = (Ellipsis, result)
        else:
            if not isinstance(args, list):
                raise TypeError("Callable[args, result]: args must be a list."
                                " Got %.100r." % (args,))
            parameters = (tuple(args), result)
        return self.__getitem_inner__(parameters)

    @_tp_cache
    def __getitem_inner__(self, parameters):
        args, result = parameters
        msg = "Callable[args, result]: result must be a type."
        result = _type_check(result, msg)
        if args is Ellipsis:
            return super(CallableMeta, self).__getitem__((_TypingEllipsis, result))
        msg = "Callable[[arg, ...], result]: each arg must be a type."
        args = tuple(_type_check(arg, msg) for arg in args)
        parameters = args + (result,)
        return super(CallableMeta, self).__getitem__(parameters)


copy._copy_dispatch[CallableMeta] = _copy_generic


class Callable(object):
    """Callable type; Callable[[int], str] is a function of (int) -> str.

    The subscription syntax must always be used with exactly two
    values: the argument list and the return type.  The argument list
    must be a list of types or ellipsis; the return type must be a single type.

    There is no syntax to indicate optional or keyword arguments,
    such function types are rarely used as callback types.
    """

    __metaclass__ = CallableMeta
    __extra__ = collections_abc.Callable
    __slots__ = ()

    def __new__(cls, *args, **kwds):
        if cls._gorg is Callable:
            raise TypeError("Type Callable cannot be instantiated; "
                            "use a non-abstract subclass instead")
        return _generic_new(cls.__next_in_mro__, cls, *args, **kwds)


def cast(typ, val):
    """Cast a value to a type.

    This returns the value unchanged.  To the type checker this
    signals that the return value has the designated type, but at
    runtime we intentionally don't check anything (we want this
    to be as fast as possible).
    """
    return val


def _get_defaults(func):
    """Internal helper to extract the default arguments, by name."""
    code = func.__code__
    pos_count = code.co_argcount
    arg_names = code.co_varnames
    arg_names = arg_names[:pos_count]
    defaults = func.__defaults__ or ()
    kwdefaults = func.__kwdefaults__
    res = dict(kwdefaults) if kwdefaults else {}
    pos_offset = pos_count - len(defaults)
    for name, value in zip(arg_names[pos_offset:], defaults):
        assert name not in res
        res[name] = value
    return res


def get_type_hints(obj, globalns=None, localns=None):
    """In Python 2 this is not supported and always returns None."""
    return None


def no_type_check(arg):
    """Decorator to indicate that annotations are not type hints.

    The argument must be a class or function; if it is a class, it
    applies recursively to all methods and classes defined in that class
    (but not to methods defined in its superclasses or subclasses).

    This mutates the function(s) or class(es) in place.
    """
    if isinstance(arg, type):
        arg_attrs = arg.__dict__.copy()
        for attr, val in arg.__dict__.items():
            if val in arg.__bases__ + (arg,):
                arg_attrs.pop(attr)
        for obj in arg_attrs.values():
            if isinstance(obj, types.FunctionType):
                obj.__no_type_check__ = True
            if isinstance(obj, type):
                no_type_check(obj)
    try:
        arg.__no_type_check__ = True
    except TypeError:  # built-in classes
        pass
    return arg


def no_type_check_decorator(decorator):
    """Decorator to give another decorator the @no_type_check effect.

    This wraps the decorator with something that wraps the decorated
    function in @no_type_check.
    """

    @functools.wraps(decorator)
    def wrapped_decorator(*args, **kwds):
        func = decorator(*args, **kwds)
        func = no_type_check(func)
        return func

    return wrapped_decorator


def _overload_dummy(*args, **kwds):
    """Helper for @overload to raise when called."""
    raise NotImplementedError(
        "You should not call an overloaded function. "
        "A series of @overload-decorated functions "
        "outside a stub module should always be followed "
        "by an implementation that is not @overload-ed.")


def overload(func):
    """Decorator for overloaded functions/methods.

    In a stub file, place two or more stub definitions for the same
    function in a row, each decorated with @overload.  For example:

      @overload
      def utf8(value: None) -> None: ...
      @overload
      def utf8(value: bytes) -> bytes: ...
      @overload
      def utf8(value: str) -> bytes: ...

    In a non-stub file (i.e. a regular .py file), do the same but
    follow it with an implementation.  The implementation should *not*
    be decorated with @overload.  For example:

      @overload
      def utf8(value: None) -> None: ...
      @overload
      def utf8(value: bytes) -> bytes: ...
      @overload
      def utf8(value: str) -> bytes: ...
      def utf8(value):
          # implementation goes here
    """
    return _overload_dummy


_PROTO_WHITELIST = ['Callable', 'Iterable', 'Iterator',
                    'Hashable', 'Sized', 'Container', 'Collection',
                    'Reversible', 'ContextManager']


class _ProtocolMeta(GenericMeta):
    """Internal metaclass for Protocol.

    This exists so Protocol classes can be generic without deriving
    from Generic.
    """
    def __init__(cls, *args, **kwargs):
        super(_ProtocolMeta, cls).__init__(*args, **kwargs)
        if not cls.__dict__.get('_is_protocol', None):
            cls._is_protocol = any(b is Protocol or
                                   isinstance(b, _ProtocolMeta) and
                                   b.__origin__ is Protocol
                                   for b in cls.__bases__)
        if cls._is_protocol:
            for base in cls.__mro__[1:]:
                if not (base in (object, Generic) or
                        base.__module__ == '_abcoll' and
                        base.__name__ in _PROTO_WHITELIST or
                        isinstance(base, TypingMeta) and base._is_protocol or
                        isinstance(base, GenericMeta) and base.__origin__ is Generic):
                    raise TypeError('Protocols can only inherit from other protocols,'
                                    ' got %r' % base)
            cls._callable_members_only = all(callable(getattr(cls, attr))
                                             for attr in cls._get_protocol_attrs())

            def _no_init(self, *args, **kwargs):
                if type(self)._is_protocol:
                    raise TypeError('Protocols cannot be instantiated')
            cls.__init__ = _no_init

        def _proto_hook(cls, other):
            if not cls.__dict__.get('_is_protocol', None):
                return NotImplemented
            if not isinstance(other, type):
                # Similar error as for issubclass(1, int)
                # (also not a chance for old-style classes)
                raise TypeError('issubclass() arg 1 must be a new-style class')
            for attr in cls._get_protocol_attrs():
                for base in other.__mro__:
                    if attr in base.__dict__:
                        if base.__dict__[attr] is None:
                            return NotImplemented
                        break
                else:
                    return NotImplemented
            return True
        if '__subclasshook__' not in cls.__dict__:
            cls.__subclasshook__ = classmethod(_proto_hook)

    def __instancecheck__(self, instance):
        # We need this method for situations where attributes are assigned in __init__
        if isinstance(instance, type):
            # This looks like a fundamental limitation of Python 2.
            # It cannot support runtime protocol metaclasses, On Python 2 classes
            # cannot be correctly inspected as instances of protocols.
            return False
        if ((not getattr(self, '_is_protocol', False) or
                self._callable_members_only) and
                issubclass(instance.__class__, self)):
            return True
        if self._is_protocol:
            if all(hasattr(instance, attr) and
                    (not callable(getattr(self, attr)) or
                     getattr(instance, attr) is not None)
                    for attr in self._get_protocol_attrs()):
                return True
        return super(GenericMeta, self).__instancecheck__(instance)

    def __subclasscheck__(self, cls):
        if (self.__dict__.get('_is_protocol', None) and
                not self.__dict__.get('_is_runtime_protocol', None)):
            if (sys._getframe(1).f_globals['__name__'] in ['abc', 'functools'] or
                    # This is needed because we remove subclasses from unions on Python 2.
                    sys._getframe(2).f_globals['__name__'] == 'typing'):
                return False
            raise TypeError("Instance and class checks can only be used with"
                            " @runtime_checkable protocols")
        if (self.__dict__.get('_is_runtime_protocol', None) and
                not self._callable_members_only):
            if sys._getframe(1).f_globals['__name__'] in ['abc', 'functools']:
                return super(GenericMeta, self).__subclasscheck__(cls)
            raise TypeError("Protocols with non-method members"
                            " don't support issubclass()")
        return super(_ProtocolMeta, self).__subclasscheck__(cls)

    def _get_protocol_attrs(self):
        attrs = set()
        for base in self.__mro__[:-1]:  # without object
            if base.__name__ in ('Protocol', 'Generic'):
                continue
            annotations = getattr(base, '__annotations__', {})
            for attr in list(base.__dict__.keys()) + list(annotations.keys()):
                if (not attr.startswith('_abc_') and attr not in (
                        '__abstractmethods__', '__annotations__', '__weakref__',
                        '_is_protocol', '_is_runtime_protocol', '__dict__',
                        '__args__', '__slots__', '_get_protocol_attrs',
                        '__next_in_mro__', '__parameters__', '__origin__',
                        '__orig_bases__', '__extra__', '__tree_hash__',
                        '__doc__', '__subclasshook__', '__init__', '__new__',
                        '__module__', '_MutableMapping__marker',
                        '__metaclass__', '_gorg', '_callable_members_only')):
                    attrs.add(attr)
        return attrs


class Protocol(object):
    """Base class for protocol classes. Protocol classes are defined as::

      class Proto(Protocol):
          def meth(self):
              # type: () -> int
              pass

    Such classes are primarily used with static type checkers that recognize
    structural subtyping (static duck-typing), for example::

      class C:
          def meth(self):
              # type: () -> int
              return 0

      def func(x):
          # type: (Proto) -> int
          return x.meth()

      func(C())  # Passes static type check

    See PEP 544 for details. Protocol classes decorated with @typing.runtime_checkable
    act as simple-minded runtime protocols that checks only the presence of
    given attributes, ignoring their type signatures.

    Protocol classes can be generic, they are defined as::

      class GenProto(Protocol[T]):
          def meth(self):
              # type: () -> T
              pass
    """

    __metaclass__ = _ProtocolMeta
    __slots__ = ()
    _is_protocol = True

    def __new__(cls, *args, **kwds):
        if cls._gorg is Protocol:
            raise TypeError("Type Protocol cannot be instantiated; "
                            "it can be used only as a base class")
        return _generic_new(cls.__next_in_mro__, cls, *args, **kwds)


def runtime_checkable(cls):
    """Mark a protocol class as a runtime protocol, so that it
    can be used with isinstance() and issubclass(). Raise TypeError
    if applied to a non-protocol class.

    This allows a simple-minded structural check very similar to the
    one-offs in collections.abc such as Hashable.
    """
    if not isinstance(cls, _ProtocolMeta) or not cls._is_protocol:
        raise TypeError('@runtime_checkable can be only applied to protocol classes,'
                        ' got %r' % cls)
    cls._is_runtime_protocol = True
    return cls


# Various ABCs mimicking those in collections.abc.
# A few are simply re-exported for completeness.

Hashable = collections_abc.Hashable  # Not generic.


class Iterable(Generic[T_co]):
    __slots__ = ()
    __extra__ = collections_abc.Iterable


class Iterator(Iterable[T_co]):
    __slots__ = ()
    __extra__ = collections_abc.Iterator


@runtime_checkable
class SupportsInt(Protocol):
    __slots__ = ()

    @abstractmethod
    def __int__(self):
        pass


@runtime_checkable
class SupportsFloat(Protocol):
    __slots__ = ()

    @abstractmethod
    def __float__(self):
        pass


@runtime_checkable
class SupportsComplex(Protocol):
    __slots__ = ()

    @abstractmethod
    def __complex__(self):
        pass


@runtime_checkable
class SupportsIndex(Protocol):
    __slots__ = ()

    @abstractmethod
    def __index__(self):
        pass


@runtime_checkable
class SupportsAbs(Protocol[T_co]):
    __slots__ = ()

    @abstractmethod
    def __abs__(self):
        pass


if hasattr(collections_abc, 'Reversible'):
    class Reversible(Iterable[T_co]):
        __slots__ = ()
        __extra__ = collections_abc.Reversible
else:
    @runtime_checkable
    class Reversible(Protocol[T_co]):
        __slots__ = ()

        @abstractmethod
        def __reversed__(self):
            pass


Sized = collections_abc.Sized  # Not generic.


class Container(Generic[T_co]):
    __slots__ = ()
    __extra__ = collections_abc.Container


# Callable was defined earlier.


class AbstractSet(Sized, Iterable[T_co], Container[T_co]):
    __slots__ = ()
    __extra__ = collections_abc.Set


class MutableSet(AbstractSet[T]):
    __slots__ = ()
    __extra__ = collections_abc.MutableSet


# NOTE: It is only covariant in the value type.
class Mapping(Sized, Iterable[KT], Container[KT], Generic[KT, VT_co]):
    __slots__ = ()
    __extra__ = collections_abc.Mapping


class MutableMapping(Mapping[KT, VT]):
    __slots__ = ()
    __extra__ = collections_abc.MutableMapping


if hasattr(collections_abc, 'Reversible'):
    class Sequence(Sized, Reversible[T_co], Container[T_co]):
        __slots__ = ()
        __extra__ = collections_abc.Sequence
else:
    class Sequence(Sized, Iterable[T_co], Container[T_co]):
        __slots__ = ()
        __extra__ = collections_abc.Sequence


class MutableSequence(Sequence[T]):
    __slots__ = ()
    __extra__ = collections_abc.MutableSequence


class ByteString(Sequence[int]):
    pass


ByteString.register(str)
ByteString.register(bytearray)


class List(list, MutableSequence[T]):
    __slots__ = ()
    __extra__ = list

    def __new__(cls, *args, **kwds):
        if cls._gorg is List:
            raise TypeError("Type List cannot be instantiated; "
                            "use list() instead")
        return _generic_new(list, cls, *args, **kwds)


class Deque(collections.deque, MutableSequence[T]):
    __slots__ = ()
    __extra__ = collections.deque

    def __new__(cls, *args, **kwds):
        if cls._gorg is Deque:
            return collections.deque(*args, **kwds)
        return _generic_new(collections.deque, cls, *args, **kwds)


class Set(set, MutableSet[T]):
    __slots__ = ()
    __extra__ = set

    def __new__(cls, *args, **kwds):
        if cls._gorg is Set:
            raise TypeError("Type Set cannot be instantiated; "
                            "use set() instead")
        return _generic_new(set, cls, *args, **kwds)


class FrozenSet(frozenset, AbstractSet[T_co]):
    __slots__ = ()
    __extra__ = frozenset

    def __new__(cls, *args, **kwds):
        if cls._gorg is FrozenSet:
            raise TypeError("Type FrozenSet cannot be instantiated; "
                            "use frozenset() instead")
        return _generic_new(frozenset, cls, *args, **kwds)


class MappingView(Sized, Iterable[T_co]):
    __slots__ = ()
    __extra__ = collections_abc.MappingView


class KeysView(MappingView[KT], AbstractSet[KT]):
    __slots__ = ()
    __extra__ = collections_abc.KeysView


class ItemsView(MappingView[Tuple[KT, VT_co]],
                AbstractSet[Tuple[KT, VT_co]],
                Generic[KT, VT_co]):
    __slots__ = ()
    __extra__ = collections_abc.ItemsView


class ValuesView(MappingView[VT_co]):
    __slots__ = ()
    __extra__ = collections_abc.ValuesView


class ContextManager(Generic[T_co]):
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


class Dict(dict, MutableMapping[KT, VT]):
    __slots__ = ()
    __extra__ = dict

    def __new__(cls, *args, **kwds):
        if cls._gorg is Dict:
            raise TypeError("Type Dict cannot be instantiated; "
                            "use dict() instead")
        return _generic_new(dict, cls, *args, **kwds)


class DefaultDict(collections.defaultdict, MutableMapping[KT, VT]):
    __slots__ = ()
    __extra__ = collections.defaultdict

    def __new__(cls, *args, **kwds):
        if cls._gorg is DefaultDict:
            return collections.defaultdict(*args, **kwds)
        return _generic_new(collections.defaultdict, cls, *args, **kwds)


class Counter(collections.Counter, Dict[T, int]):
    __slots__ = ()
    __extra__ = collections.Counter

    def __new__(cls, *args, **kwds):
        if cls._gorg is Counter:
            return collections.Counter(*args, **kwds)
        return _generic_new(collections.Counter, cls, *args, **kwds)


# Determine what base class to use for Generator.
if hasattr(collections_abc, 'Generator'):
    # Sufficiently recent versions of 3.5 have a Generator ABC.
    _G_base = collections_abc.Generator
else:
    # Fall back on the exact type.
    _G_base = types.GeneratorType


class Generator(Iterator[T_co], Generic[T_co, T_contra, V_co]):
    __slots__ = ()
    __extra__ = _G_base

    def __new__(cls, *args, **kwds):
        if cls._gorg is Generator:
            raise TypeError("Type Generator cannot be instantiated; "
                            "create a subclass instead")
        return _generic_new(_G_base, cls, *args, **kwds)


# Internal type variable used for Type[].
CT_co = TypeVar('CT_co', covariant=True, bound=type)


# This is not a real generic class.  Don't use outside annotations.
class Type(Generic[CT_co]):
    """A special construct usable to annotate class objects.

    For example, suppose we have the following classes::

      class User: ...  # Abstract base for User classes
      class BasicUser(User): ...
      class ProUser(User): ...
      class TeamUser(User): ...

    And a function that takes a class argument that's a subclass of
    User and returns an instance of the corresponding class::

      U = TypeVar('U', bound=User)
      def new_user(user_class: Type[U]) -> U:
          user = user_class()
          # (Here we could write the user object to a database)
          return user

      joe = new_user(BasicUser)

    At this point the type checker knows that joe has type BasicUser.
    """
    __slots__ = ()
    __extra__ = type


def NamedTuple(typename, fields):
    """Typed version of namedtuple.

    Usage::

        Employee = typing.NamedTuple('Employee', [('name', str), ('id', int)])

    This is equivalent to::

        Employee = collections.namedtuple('Employee', ['name', 'id'])

    The resulting class has one extra attribute: _field_types,
    giving a dict mapping field names to types.  (The field names
    are in the _fields attribute, which is part of the namedtuple
    API.)
    """
    fields = [(n, t) for n, t in fields]
    cls = collections.namedtuple(typename, [n for n, t in fields])
    cls._field_types = dict(fields)
    # Set the module to the caller's module (otherwise it'd be 'typing').
    try:
        cls.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass
    return cls


def _check_fails(cls, other):
    try:
        if sys._getframe(1).f_globals['__name__'] not in ['abc', 'functools', 'typing']:
            # Typed dicts are only for static structural subtyping.
            raise TypeError('TypedDict does not support instance and class checks')
    except (AttributeError, ValueError):
        pass
    return False


def _dict_new(cls, *args, **kwargs):
    return dict(*args, **kwargs)


def _typeddict_new(cls, _typename, _fields=None, **kwargs):
    total = kwargs.pop('total', True)
    if _fields is None:
        _fields = kwargs
    elif kwargs:
        raise TypeError("TypedDict takes either a dict or keyword arguments,"
                        " but not both")

    ns = {'__annotations__': dict(_fields), '__total__': total}
    try:
        # Setting correct module is necessary to make typed dict classes pickleable.
        ns['__module__'] = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return _TypedDictMeta(_typename, (), ns)


class _TypedDictMeta(type):
    def __new__(cls, name, bases, ns, total=True):
        # Create new typed dict class object.
        # This method is called directly when TypedDict is subclassed,
        # or via _typeddict_new when TypedDict is instantiated. This way
        # TypedDict supports all three syntaxes described in its docstring.
        # Subclasses and instances of TypedDict return actual dictionaries
        # via _dict_new.
        ns['__new__'] = _typeddict_new if name == b'TypedDict' else _dict_new
        tp_dict = super(_TypedDictMeta, cls).__new__(cls, name, (dict,), ns)

        anns = ns.get('__annotations__', {})
        msg = "TypedDict('Name', {f0: t0, f1: t1, ...}); each t must be a type"
        anns = {n: _type_check(tp, msg) for n, tp in anns.items()}
        for base in bases:
            anns.update(base.__dict__.get('__annotations__', {}))
        tp_dict.__annotations__ = anns
        if not hasattr(tp_dict, '__total__'):
            tp_dict.__total__ = total
        return tp_dict

    __instancecheck__ = __subclasscheck__ = _check_fails


TypedDict = _TypedDictMeta(b'TypedDict', (dict,), {})
TypedDict.__module__ = __name__
TypedDict.__doc__ = \
    """A simple typed name space. At runtime it is equivalent to a plain dict.

    TypedDict creates a dictionary type that expects all of its
    instances to have a certain set of keys, with each key
    associated with a value of a consistent type. This expectation
    is not checked at runtime but is only enforced by type checkers.
    Usage::

        Point2D = TypedDict('Point2D', {'x': int, 'y': int, 'label': str})

        a: Point2D = {'x': 1, 'y': 2, 'label': 'good'}  # OK
        b: Point2D = {'z': 3, 'label': 'bad'}           # Fails type check

        assert Point2D(x=1, y=2, label='first') == dict(x=1, y=2, label='first')

    The type info could be accessed via Point2D.__annotations__. TypedDict
    supports an additional equivalent form::

        Point2D = TypedDict('Point2D', x=int, y=int, label=str)
    """


def NewType(name, tp):
    """NewType creates simple unique types with almost zero
    runtime overhead. NewType(name, tp) is considered a subtype of tp
    by static type checkers. At runtime, NewType(name, tp) returns
    a dummy function that simply returns its argument. Usage::

        UserId = NewType('UserId', int)

        def name_by_id(user_id):
            # type: (UserId) -> str
            ...

        UserId('user')          # Fails type check

        name_by_id(42)          # Fails type check
        name_by_id(UserId(42))  # OK

        num = UserId(5) + 1     # type: int
    """

    def new_type(x):
        return x

    # Some versions of Python 2 complain because of making all strings unicode
    new_type.__name__ = str(name)
    new_type.__supertype__ = tp
    return new_type


# Python-version-specific alias (Python 2: unicode; Python 3: str)
Text = unicode


# Constant that's True when type checking, but False here.
TYPE_CHECKING = False


class IO(Generic[AnyStr]):
    """Generic base class for TextIO and BinaryIO.

    This is an abstract, generic version of the return of open().

    NOTE: This does not distinguish between the different possible
    classes (text vs. binary, read vs. write vs. read/write,
    append-only, unbuffered).  The TextIO and BinaryIO subclasses
    below capture the distinctions between text vs. binary, which is
    pervasive in the interface; however we currently do not offer a
    way to track the other distinctions in the type system.
    """

    __slots__ = ()

    @abstractproperty
    def mode(self):
        pass

    @abstractproperty
    def name(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractproperty
    def closed(self):
        pass

    @abstractmethod
    def fileno(self):
        pass

    @abstractmethod
    def flush(self):
        pass

    @abstractmethod
    def isatty(self):
        pass

    @abstractmethod
    def read(self, n=-1):
        pass

    @abstractmethod
    def readable(self):
        pass

    @abstractmethod
    def readline(self, limit=-1):
        pass

    @abstractmethod
    def readlines(self, hint=-1):
        pass

    @abstractmethod
    def seek(self, offset, whence=0):
        pass

    @abstractmethod
    def seekable(self):
        pass

    @abstractmethod
    def tell(self):
        pass

    @abstractmethod
    def truncate(self, size=None):
        pass

    @abstractmethod
    def writable(self):
        pass

    @abstractmethod
    def write(self, s):
        pass

    @abstractmethod
    def writelines(self, lines):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, type, value, traceback):
        pass


class BinaryIO(IO[bytes]):
    """Typed version of the return of open() in binary mode."""

    __slots__ = ()

    @abstractmethod
    def write(self, s):
        pass

    @abstractmethod
    def __enter__(self):
        pass


class TextIO(IO[unicode]):
    """Typed version of the return of open() in text mode."""

    __slots__ = ()

    @abstractproperty
    def buffer(self):
        pass

    @abstractproperty
    def encoding(self):
        pass

    @abstractproperty
    def errors(self):
        pass

    @abstractproperty
    def line_buffering(self):
        pass

    @abstractproperty
    def newlines(self):
        pass

    @abstractmethod
    def __enter__(self):
        pass


class io(object):
    """Wrapper namespace for IO generic classes."""

    __all__ = ['IO', 'TextIO', 'BinaryIO']
    IO = IO
    TextIO = TextIO
    BinaryIO = BinaryIO


io.__name__ = __name__ + b'.io'
sys.modules[io.__name__] = io


Pattern = _TypeAlias('Pattern', AnyStr, type(stdlib_re.compile('')),
                     lambda p: p.pattern)
Match = _TypeAlias('Match', AnyStr, type(stdlib_re.match('', '')),
                   lambda m: m.re.pattern)


class re(object):
    """Wrapper namespace for re type aliases."""

    __all__ = ['Pattern', 'Match']
    Pattern = Pattern
    Match = Match


re.__name__ = __name__ + b'.re'
sys.modules[re.__name__] = re
