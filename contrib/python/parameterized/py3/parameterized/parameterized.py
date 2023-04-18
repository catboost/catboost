import re
import sys
import inspect
import warnings
from typing import Iterable
from functools import wraps
from types import MethodType as MethodType
from collections import namedtuple

try:
    from unittest import mock
except ImportError:
    try:
        import mock
    except ImportError:
        mock = None

try:
    from collections import OrderedDict as MaybeOrderedDict
except ImportError:
    MaybeOrderedDict = dict

from unittest import TestCase

try:
    from unittest import SkipTest
except ImportError:
    class SkipTest(Exception):
        pass

# NOTE: even though Python 2 support has been dropped, these checks have been
# left in place to avoid merge conflicts. They can be removed in the future, and
# future code can be written to assume Python 3.
PY3 = sys.version_info[0] == 3
PY2 = sys.version_info[0] == 2


if PY3:
    # Python 3 doesn't have an InstanceType, so just use a dummy type.
    class InstanceType():
        pass
    lzip = lambda *a: list(zip(*a))
    text_type = str
    string_types = str,
    bytes_type = bytes
    def make_method(func, instance, type):
        if instance is None:
            return func
        return MethodType(func, instance)
else:
    from types import InstanceType
    lzip = zip
    text_type = unicode
    bytes_type = str
    string_types = basestring,
    def make_method(func, instance, type):
        return MethodType(func, instance, type)

def to_text(x):
    if isinstance(x, text_type):
        return x
    try:
        return text_type(x, "utf-8")
    except UnicodeDecodeError:
        return text_type(x, "latin1")

CompatArgSpec = namedtuple("CompatArgSpec", "args varargs keywords defaults")


def getargspec(func):
    if PY2:
        return CompatArgSpec(*inspect.getargspec(func))
    args = inspect.getfullargspec(func)
    if args.kwonlyargs:
        raise TypeError((
            "parameterized does not (yet) support functions with keyword "
            "only arguments, but %r has keyword only arguments. "
            "Please open an issue with your usecase if this affects you: "
            "https://github.com/wolever/parameterized/issues/new"
        ) %(func, ))
    return CompatArgSpec(*args[:4])


def skip_on_empty_helper(*a, **kw):
    raise SkipTest("parameterized input is empty")


def reapply_patches_if_need(func):

    def dummy_wrapper(orgfunc):
        @wraps(orgfunc)
        def dummy_func(*args, **kwargs):
            return orgfunc(*args, **kwargs)
        return dummy_func

    if hasattr(func, 'patchings'):
        is_original_async = inspect.iscoroutinefunction(func)
        func = dummy_wrapper(func)
        tmp_patchings = func.patchings
        delattr(func, 'patchings')
        for patch_obj in tmp_patchings:
            if is_original_async:
                func = patch_obj.decorate_async_callable(func)
            else:
                func = patch_obj.decorate_callable(func)
    return func


# `parameterized.expand` strips out `mock` patches from the source method in favor of re-applying them over the
# generated methods instead. Sadly, this can cause problems with old versions of the `mock` package, as shown in
# https://bugs.python.org/issue40126 (bpo-40126).
#
# Long story short, bpo-40126 arises whenever the `patchings` list of a `mock`-decorated method is left fully empty.
#
# The bug has been fixed in the `mock` code itself since:
#   - Python 3.7.8-rc1, 3.8.3-rc1 and later (for the `unittest.mock` package) [0][1].
#   - Version 4 of the `mock` backport package (https://pypi.org/project/mock/) [2].
#
# To work around the problem when running old `mock` versions, we avoid fully stripping out patches from the source
# method in favor of replacing them with a "dummy" no-op patch instead.
#
# [0] https://docs.python.org/release/3.7.10/whatsnew/changelog.html#python-3-7-8-release-candidate-1
# [1] https://docs.python.org/release/3.8.10/whatsnew/changelog.html#python-3-8-3-release-candidate-1
# [2] https://mock.readthedocs.io/en/stable/changelog.html#b1

PYTHON_DOESNT_HAVE_FIX_FOR_BPO_40126 = (
    sys.version_info[:3] < (3, 7, 8) or (sys.version_info[:2] >= (3, 8) and sys.version_info[:3] < (3, 8, 3))
)

try:
    import mock as _mock_backport
except ImportError:
    _mock_backport = None

MOCK_BACKPORT_DOESNT_HAVE_FIX_FOR_BPO_40126 = _mock_backport is not None and _mock_backport.version_info[0] < 4

AVOID_CLEARING_MOCK_PATCHES = PYTHON_DOESNT_HAVE_FIX_FOR_BPO_40126 or MOCK_BACKPORT_DOESNT_HAVE_FIX_FOR_BPO_40126


class DummyPatchTarget(object):
    dummy_attribute = None

    @staticmethod
    def create_dummy_patch():
        if mock is not None:
            return mock.patch.object(DummyPatchTarget(), "dummy_attribute", new=None)
        else:
            raise ImportError("Missing mock package")


def delete_patches_if_need(func):
    if hasattr(func, 'patchings'):
        if AVOID_CLEARING_MOCK_PATCHES:
            func.patchings[:] = [DummyPatchTarget.create_dummy_patch()]
        else:
            func.patchings[:] = []


_param = namedtuple("param", "args kwargs")

class param(_param):
    """ Represents a single parameter to a test case.

        For example::

            >>> p = param("foo", bar=16)
            >>> p
            param("foo", bar=16)
            >>> p.args
            ('foo', )
            >>> p.kwargs
            {'bar': 16}

        Intended to be used as an argument to ``@parameterized``::

            @parameterized([
                param("foo", bar=16),
            ])
            def test_stuff(foo, bar=16):
                pass
        """

    def __new__(cls, *args , **kwargs):
        return _param.__new__(cls, args, kwargs)

    @classmethod
    def explicit(cls, args=None, kwargs=None):
        """ Creates a ``param`` by explicitly specifying ``args`` and
            ``kwargs``::

                >>> param.explicit([1,2,3])
                param(*(1, 2, 3))
                >>> param.explicit(kwargs={"foo": 42})
                param(*(), **{"foo": "42"})
            """
        args = args or ()
        kwargs = kwargs or {}
        return cls(*args, **kwargs)

    @classmethod
    def from_decorator(cls, args):
        """ Returns an instance of ``param()`` for ``@parameterized`` argument
            ``args``::

                >>> param.from_decorator((42, ))
                param(args=(42, ), kwargs={})
                >>> param.from_decorator("foo")
                param(args=("foo", ), kwargs={})
            """
        if isinstance(args, param):
            return args
        elif isinstance(args, (str, bytes)) or not isinstance(args, Iterable):
            args = (args, )
        try:
            return cls(*args)
        except TypeError as e:
            if "after * must be" not in str(e):
                raise
            raise TypeError(
                "Parameters must be tuples, but %r is not (hint: use '(%r, )')"
                %(args, args),
            )

    def __repr__(self):
        return "param(*%r, **%r)" %self


class QuietOrderedDict(MaybeOrderedDict):
    """ When OrderedDict is available, use it to make sure that the kwargs in
        doc strings are consistently ordered. """
    __str__ = dict.__str__
    __repr__ = dict.__repr__


def parameterized_argument_value_pairs(func, p):
    """Return tuples of parameterized arguments and their values.

        This is useful if you are writing your own doc_func
        function and need to know the values for each parameter name::

            >>> def func(a, foo=None, bar=42, **kwargs): pass
            >>> p = param(1, foo=7, extra=99)
            >>> parameterized_argument_value_pairs(func, p)
            [("a", 1), ("foo", 7), ("bar", 42), ("**kwargs", {"extra": 99})]

        If the function's first argument is named ``self`` then it will be
        ignored::

            >>> def func(self, a): pass
            >>> p = param(1)
            >>> parameterized_argument_value_pairs(func, p)
            [("a", 1)]

        Additionally, empty ``*args`` or ``**kwargs`` will be ignored::

            >>> def func(foo, *args): pass
            >>> p = param(1)
            >>> parameterized_argument_value_pairs(func, p)
            [("foo", 1)]
            >>> p = param(1, 16)
            >>> parameterized_argument_value_pairs(func, p)
            [("foo", 1), ("*args", (16, ))]
    """
    argspec = getargspec(func)
    arg_offset = 1 if argspec.args[:1] == ["self"] else 0

    named_args = argspec.args[arg_offset:]

    result = lzip(named_args, p.args)
    named_args = argspec.args[len(result) + arg_offset:]
    varargs = p.args[len(result):]

    result.extend([
        (name, p.kwargs.get(name, default))
        for (name, default)
        in zip(named_args, argspec.defaults or [])
    ])

    seen_arg_names = set([ n for (n, _) in result ])
    keywords = QuietOrderedDict(sorted([
        (name, p.kwargs[name])
        for name in p.kwargs
        if name not in seen_arg_names
    ]))

    if varargs:
        result.append(("*%s" %(argspec.varargs, ), tuple(varargs)))

    if keywords:
        result.append(("**%s" %(argspec.keywords, ), keywords))

    return result


def short_repr(x, n=64):
    """ A shortened repr of ``x`` which is guaranteed to be ``unicode``::

            >>> short_repr("foo")
            u"foo"
            >>> short_repr("123456789", n=4)
            u"12...89"
    """

    x_repr = to_text(repr(x))
    if len(x_repr) > n:
        x_repr = x_repr[:n//2] + "..." + x_repr[len(x_repr) - n//2:]
    return x_repr


def default_doc_func(func, num, p):
    if func.__doc__ is None:
        return None

    all_args_with_values = parameterized_argument_value_pairs(func, p)

    # Assumes that the function passed is a bound method.
    descs = ["%s=%s" %(n, short_repr(v)) for n, v in all_args_with_values]

    # The documentation might be a multiline string, so split it
    # and just work with the first string, ignoring the period
    # at the end if there is one.
    first, nl, rest = func.__doc__.lstrip().partition("\n")
    suffix = ""
    if first.endswith("."):
        suffix = "."
        first = first[:-1]
    args = "%s[with %s]" %(len(first) and " " or "", ", ".join(descs))
    return "".join(
        to_text(x)
        for x in [first.rstrip(), args, suffix, nl, rest]
    )


def default_name_func(func, num, p):
    base_name = func.__name__
    name_suffix = "_%s" %(num, )

    if len(p.args) > 0 and isinstance(p.args[0], string_types):
        name_suffix += "_" + parameterized.to_safe_name(p.args[0])
    return base_name + name_suffix


_test_runner_override = None
_test_runner_guess = False
_test_runners = set(["unittest", "unittest2", "nose", "nose2", "pytest"])
_test_runner_aliases = {
    "_pytest": "pytest",
}


def set_test_runner(name):
    global _test_runner_override
    if name not in _test_runners:
        raise TypeError(
            "Invalid test runner: %r (must be one of: %s)"
            %(name, ", ".join(_test_runners)),
        )
    _test_runner_override = name


def detect_runner():
    """ Guess which test runner we're using by traversing the stack and looking
        for the first matching module. This *should* be reasonably safe, as
        it's done during test discovery where the test runner should be the
        stack frame immediately outside. """
    if _test_runner_override is not None:
        return _test_runner_override
    global _test_runner_guess
    if _test_runner_guess is False:
        stack = inspect.stack()
        for record in reversed(stack):
            frame = record[0]
            module = frame.f_globals.get("__name__").partition(".")[0]
            if module in _test_runner_aliases:
                module = _test_runner_aliases[module]
            if module in _test_runners:
                _test_runner_guess = module
                break
            if record[1].endswith("python2.6/unittest.py"):
                _test_runner_guess = "unittest"
                break
        else:
            _test_runner_guess = None
    return _test_runner_guess



class parameterized(object):
    """ Parameterize a test case::

            class TestInt(object):
                @parameterized([
                    ("A", 10),
                    ("F", 15),
                    param("10", 42, base=42)
                ])
                def test_int(self, input, expected, base=16):
                    actual = int(input, base=base)
                    assert_equal(actual, expected)

            @parameterized([
                (2, 3, 5)
                (3, 5, 8),
            ])
            def test_add(a, b, expected):
                assert_equal(a + b, expected)
        """

    def __init__(self, input, doc_func=None, skip_on_empty=False):
        self.get_input = self.input_as_callable(input)
        self.doc_func = doc_func or default_doc_func
        self.skip_on_empty = skip_on_empty

    def __call__(self, test_func):
        self.assert_not_in_testcase_subclass()

        @wraps(test_func)
        def wrapper(test_self=None):
            test_cls = test_self and type(test_self)
            if test_self is not None:
                if issubclass(test_cls, InstanceType):
                    raise TypeError((
                        "@parameterized can't be used with old-style classes, but "
                        "%r has an old-style class. Consider using a new-style "
                        "class, or '@parameterized.expand' "
                        "(see http://stackoverflow.com/q/54867/71522 for more "
                        "information on old-style classes)."
                    ) %(test_self, ))

            original_doc = wrapper.__doc__
            for num, args in enumerate(wrapper.parameterized_input):
                p = param.from_decorator(args)
                unbound_func, nose_tuple = self.param_as_nose_tuple(test_self, test_func, num, p)
                try:
                    wrapper.__doc__ = nose_tuple[0].__doc__
                    # Nose uses `getattr(instance, test_func.__name__)` to get
                    # a method bound to the test instance (as opposed to a
                    # method bound to the instance of the class created when
                    # tests were being enumerated). Set a value here to make
                    # sure nose can get the correct test method.
                    if test_self is not None:
                        setattr(test_cls, test_func.__name__, unbound_func)
                    yield nose_tuple
                finally:
                    if test_self is not None:
                        delattr(test_cls, test_func.__name__)
                    wrapper.__doc__ = original_doc

        input = self.get_input()
        if not input:
            if not self.skip_on_empty:
                raise ValueError(
                    "Parameters iterable is empty (hint: use "
                    "`parameterized([], skip_on_empty=True)` to skip "
                    "this test when the input is empty)"
                )
            wrapper = wraps(test_func)(skip_on_empty_helper)

        wrapper.parameterized_input = input
        wrapper.parameterized_func = test_func
        test_func.__name__ = "_parameterized_original_%s" %(test_func.__name__, )

        return wrapper

    def param_as_nose_tuple(self, test_self, func, num, p):
        nose_func = wraps(func)(lambda *args: func(*args[:-1], **args[-1]))
        nose_func.__doc__ = self.doc_func(func, num, p)
        # Track the unbound function because we need to setattr the unbound
        # function onto the class for nose to work (see comments above), and
        # Python 3 doesn't let us pull the function out of a bound method.
        unbound_func = nose_func
        if test_self is not None:
            # Under nose on Py2 we need to return an unbound method to make
            # sure that the `self` in the method is properly shared with the
            # `self` used in `setUp` and `tearDown`. But only there. Everyone
            # else needs a bound method.
            func_self = (
                None if PY2 and detect_runner() == "nose" else
                test_self
            )
            nose_func = make_method(nose_func, func_self, type(test_self))
        return unbound_func, (nose_func, ) + p.args + (p.kwargs or {}, )

    def assert_not_in_testcase_subclass(self):
        parent_classes = self._terrible_magic_get_defining_classes()
        if any(issubclass(cls, TestCase) for cls in parent_classes):
            raise Exception("Warning: '@parameterized' tests won't work "
                            "inside subclasses of 'TestCase' - use "
                            "'@parameterized.expand' instead.")

    def _terrible_magic_get_defining_classes(self):
        """ Returns the set of parent classes of the class currently being defined.
            Will likely only work if called from the ``parameterized`` decorator.
            This function is entirely @brandon_rhodes's fault, as he suggested
            the implementation: http://stackoverflow.com/a/8793684/71522
            """
        stack = inspect.stack()
        if len(stack) <= 4:
            return []
        frame = stack[4]
        code_context = frame[4] and frame[4][0].strip()
        if not (code_context and code_context.startswith("class ")):
            return []
        _, _, parents = code_context.partition("(")
        parents, _, _ = parents.partition(")")
        return eval("[" + parents + "]", frame[0].f_globals, frame[0].f_locals)

    @classmethod
    def input_as_callable(cls, input):
        if callable(input):
            return lambda: cls.check_input_values(input())
        input_values = cls.check_input_values(input)
        return lambda: input_values

    @classmethod
    def check_input_values(cls, input_values):
        # Explicitly convery non-list inputs to a list so that:
        # 1. A helpful exception will be raised if they aren't iterable, and
        # 2. Generators are unwrapped exactly once (otherwise `nosetests
        #    --processes=n` has issues; see:
        #    https://github.com/wolever/nose-parameterized/pull/31)
        if not isinstance(input_values, list):
            input_values = list(input_values)
        return [ param.from_decorator(p) for p in input_values ]

    @classmethod
    def expand(cls, input, name_func=None, doc_func=None, skip_on_empty=False,
               namespace=None, **legacy):
        """ A "brute force" method of parameterizing test cases. Creates new
            test cases and injects them into the namespace that the wrapped
            function is being defined in. Useful for parameterizing tests in
            subclasses of 'UnitTest', where Nose test generators don't work.

            :param input: An iterable of values to pass to the test function.
            :param name_func: A function that takes a single argument (the
                value from the input iterable) and returns a string to use as
                the name of the test case. If not provided, the name of the
                test case will be the name of the test function with the
                parameter value appended.
            :param doc_func: A function that takes a single argument (the
                value from the input iterable) and returns a string to use as
                the docstring of the test case. If not provided, the docstring
                of the test case will be the docstring of the test function.
            :param skip_on_empty: If True, the test will be skipped if the
                input iterable is empty. If False, a ValueError will be raised
                if the input iterable is empty.
            :param namespace: The namespace (dict-like) to inject the test cases
                into. If not provided, the namespace of the test function will
                be used.

            >>> @parameterized.expand([("foo", 1, 2)])
            ... def test_add1(name, input, expected):
            ...     actual = add1(input)
            ...     assert_equal(actual, expected)
            ...
            >>> locals()
            ... 'test_add1_foo_0': <function ...> ...
            >>>
            """

        if "testcase_func_name" in legacy:
            warnings.warn("testcase_func_name= is deprecated; use name_func=",
                          DeprecationWarning, stacklevel=2)
            if not name_func:
                name_func = legacy["testcase_func_name"]

        if "testcase_func_doc" in legacy:
            warnings.warn("testcase_func_doc= is deprecated; use doc_func=",
                          DeprecationWarning, stacklevel=2)
            if not doc_func:
                doc_func = legacy["testcase_func_doc"]

        doc_func = doc_func or default_doc_func
        name_func = name_func or default_name_func

        def parameterized_expand_wrapper(f, instance=None):
            frame_locals = namespace
            if frame_locals is None:
                frame_locals = inspect.currentframe().f_back.f_locals

            parameters = cls.input_as_callable(input)()

            if not parameters:
                if not skip_on_empty:
                    raise ValueError(
                        "Parameters iterable is empty (hint: use "
                        "`parameterized.expand([], skip_on_empty=True)` to skip "
                        "this test when the input is empty)"
                    )
                return wraps(f)(skip_on_empty_helper)

            digits = len(str(len(parameters) - 1))
            for num, p in enumerate(parameters):
                name = name_func(f, "{num:0>{digits}}".format(digits=digits, num=num), p)
                # If the original function has patches applied by 'mock.patch',
                # re-construct all patches on the just former decoration layer
                # of param_as_standalone_func so as not to share
                # patch objects between new functions
                nf = reapply_patches_if_need(f)
                frame_locals[name] = cls.param_as_standalone_func(p, nf, name)
                frame_locals[name].__doc__ = doc_func(f, num, p)

            # Delete original patches to prevent new function from evaluating
            # original patching object as well as re-constructed patches.
            delete_patches_if_need(f)

            f.__test__ = False

        return parameterized_expand_wrapper

    @classmethod
    def param_as_standalone_func(cls, p, func, name):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def standalone_func(*a, **kw):
                return await func(*(a + p.args), **p.kwargs, **kw)
        else:
            @wraps(func)
            def standalone_func(*a, **kw):
                return func(*(a + p.args), **p.kwargs, **kw)

        standalone_func.__name__ = name

        # place_as is used by py.test to determine what source file should be
        # used for this test.
        standalone_func.place_as = func

        # Remove __wrapped__ because py.test will try to look at __wrapped__
        # to determine which parameters should be used with this test case,
        # and obviously we don't need it to do any parameterization.
        try:
            del standalone_func.__wrapped__
        except AttributeError:
            pass
        return standalone_func

    @classmethod
    def to_safe_name(cls, s):
        if not isinstance(s, str):
            s = str(s)
        return str(re.sub("[^a-zA-Z0-9_]+", "_", s))


def parameterized_class(attrs, input_values=None, class_name_func=None, classname_func=None):
    """ Parameterizes a test class by setting attributes on the class.

        Can be used in two ways:

        1) With a list of dictionaries containing attributes to override::

            @parameterized_class([
                { "username": "foo" },
                { "username": "bar", "access_level": 2 },
            ])
            class TestUserAccessLevel(TestCase):
                ...

        2) With a tuple of attributes, then a list of tuples of values:

            @parameterized_class(("username", "access_level"), [
                ("foo", 1),
                ("bar", 2)
            ])
            class TestUserAccessLevel(TestCase):
                ...

    """

    if isinstance(attrs, string_types):
        attrs = [attrs]

    input_dicts = (
        attrs if input_values is None else
        [dict(zip(attrs, vals)) for vals in input_values]
    )

    class_name_func = class_name_func or default_class_name_func

    if classname_func:
        warnings.warn(
            "classname_func= is deprecated; use class_name_func= instead. "
            "See: https://github.com/wolever/parameterized/pull/74#issuecomment-613577057",
            DeprecationWarning,
            stacklevel=2,
        )
        class_name_func = lambda cls, idx, input: classname_func(cls, idx, input_dicts)

    def decorator(base_class):
        test_class_module = sys.modules[base_class.__module__].__dict__
        for idx, input_dict in enumerate(input_dicts):
            test_class_dict = dict(base_class.__dict__)
            test_class_dict.update(input_dict)

            name = class_name_func(base_class, idx, input_dict)

            test_class_module[name] = type(name, (base_class, ), test_class_dict)

        # We need to leave the base class in place (see issue #73), but if we
        # leave the test_ methods in place, the test runner will try to pick
        # them up and run them... which doesn't make sense, since no parameters
        # will have been applied.
        # Address this by iterating over the base class and remove all test
        # methods.
        for method_name in list(base_class.__dict__):
            if method_name.startswith("test"):
                delattr(base_class, method_name)
        return base_class

    return decorator


def get_class_name_suffix(params_dict):
    if "name" in params_dict:
        return parameterized.to_safe_name(params_dict["name"])

    params_vals = (
        params_dict.values() if PY3 else
        (v for (_, v) in sorted(params_dict.items()))
    )
    return parameterized.to_safe_name(next((
        v for v in params_vals
        if isinstance(v, string_types)
    ), ""))


def default_class_name_func(cls, num, params_dict):
    suffix = get_class_name_suffix(params_dict)
    return "%s_%s%s" %(
        cls.__name__,
        num,
        suffix and "_" + suffix,
    )
