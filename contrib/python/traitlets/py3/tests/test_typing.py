from __future__ import annotations

import logging
import typing as t
from abc import ABC

import pytest

from traitlets import (
    Any,
    Bool,
    CInt,
    Dict,
    Enum,
    HasTraits,
    Instance,
    Int,
    List,
    Set,
    TCPAddress,
    Type,
    Unicode,
    Union,
    default,
    observe,
    validate,
)
from traitlets.config import Config

if not t.TYPE_CHECKING:

    def reveal_type(*args: t.Any, **kwargs: t.Any) -> None:
        pass


# mypy: disallow-untyped-calls


class Foo:
    def __init__(self, c: t.Any) -> None:
        self.c = c


@pytest.mark.mypy_testing
def mypy_decorator_typing() -> None:
    class T(HasTraits):
        foo = Unicode("").tag(config=True)

        @default("foo")
        def _default_foo(self) -> str:
            return "hi"

        @observe("foo")
        def _foo_observer(self, change: t.Any) -> bool:
            return True

        @validate("foo")
        def _foo_validate(self, commit: t.Any) -> bool:
            return True

    t = T()
    reveal_type(t.foo)  # R: builtins.str
    reveal_type(t._foo_observer)  # R: Any
    reveal_type(t._foo_validate)  # R: Any


@pytest.mark.mypy_testing
def mypy_config_typing() -> None:
    c = Config(
        {
            "ExtractOutputPreprocessor": {"enabled": True},
        }
    )
    reveal_type(c)  # R: traitlets.config.loader.Config


@pytest.mark.mypy_testing
def mypy_union_typing() -> None:
    class T(HasTraits):
        style = Union(
            [Unicode("default"), Type(klass=object)],
            help="Name of the pygments style to use",
            default_value="hi",
        ).tag(config=True)

    t = T()
    reveal_type(Union("foo"))  # R: traitlets.traitlets.Union
    reveal_type(Union("").tag(sync=True))  # R: traitlets.traitlets.Union
    reveal_type(Union(None, allow_none=True))  # R: traitlets.traitlets.Union
    reveal_type(Union(None, allow_none=True).tag(sync=True))  # R: traitlets.traitlets.Union
    reveal_type(T.style)  # R: traitlets.traitlets.Union
    reveal_type(t.style)  # R: Any


@pytest.mark.mypy_testing
def mypy_list_typing() -> None:
    class T(HasTraits):
        latex_command = List(
            ["xelatex", "{filename}", "-quiet"], help="Shell command used to compile latex."
        ).tag(config=True)

    t = T()
    reveal_type(List(["foo"]))  # R: traitlets.traitlets.List[builtins.str]
    reveal_type(List([""]).tag(sync=True))  # R: traitlets.traitlets.List[builtins.str]
    reveal_type(List(None, allow_none=True))  # R: traitlets.traitlets.List[Never]
    reveal_type(
        List(None, allow_none=True).tag(sync=True)  # R: traitlets.traitlets.List[Never]
    )
    reveal_type(T.latex_command)  # R: traitlets.traitlets.List[builtins.str]
    reveal_type(t.latex_command)  # R: builtins.list[builtins.str]


@pytest.mark.mypy_testing
def mypy_dict_typing() -> None:
    class T(HasTraits):
        foo = Dict({}, help="Shell command used to compile latex.").tag(config=True)

    t = T()
    reveal_type(Dict(None, allow_none=True))  # R: traitlets.traitlets.Dict[builtins.str, Any]
    reveal_type(
        Dict(None, allow_none=True).tag(sync=True)  # R: traitlets.traitlets.Dict[builtins.str, Any]
    )
    reveal_type(T.foo)  # R: traitlets.traitlets.Dict[builtins.str, Any]
    reveal_type(t.foo)  # R: builtins.dict[builtins.str, Any]


@pytest.mark.mypy_testing
def mypy_type_typing() -> None:
    class KernelSpec:
        item = Unicode("foo")

    class KernelSpecSubclass(KernelSpec):
        other = Unicode("bar")

    class GatewayTokenRenewerBase(ABC):
        item = Unicode("foo")

    class KernelSpecManager(HasTraits):
        """A manager for kernel specs."""

        kernel_spec_class = Type(
            KernelSpec,
            config=True,
            help="""The kernel spec class.  This is configurable to allow
            subclassing of the KernelSpecManager for customized behavior.
            """,
        )
        other_class = Type("foo.bar.baz")

        other_kernel_spec_class = Type(
            default_value=KernelSpecSubclass,
            klass=KernelSpec,
            config=True,
        )

        gateway_token_renewer_class = Type(
            klass=GatewayTokenRenewerBase,
            config=True,
            help="""The class to use for Gateway token renewal. (JUPYTER_GATEWAY_TOKEN_RENEWER_CLASS env var)""",
        )

    t = KernelSpecManager()
    reveal_type(t.kernel_spec_class)  # R: def () -> tests.test_typing.KernelSpec@129
    reveal_type(t.kernel_spec_class())  # R: tests.test_typing.KernelSpec@129
    reveal_type(t.kernel_spec_class().item)  # R: builtins.str
    reveal_type(t.other_class)  # R: builtins.type
    reveal_type(t.other_class())  # R: Any
    reveal_type(t.other_kernel_spec_class)  # R: def () -> tests.test_typing.KernelSpec@129
    reveal_type(t.other_kernel_spec_class())  # R: tests.test_typing.KernelSpec@129
    reveal_type(
        t.gateway_token_renewer_class  # R: def () -> tests.test_typing.GatewayTokenRenewerBase@135
    )
    reveal_type(t.gateway_token_renewer_class())  # R: tests.test_typing.GatewayTokenRenewerBase@135


@pytest.mark.mypy_testing
def mypy_unicode_typing() -> None:
    class T(HasTraits):
        export_format = Unicode(
            allow_none=False,
            help="""The export format to be used, either one of the built-in formats
            or a dotted object name that represents the import path for an
            ``Exporter`` class""",
        ).tag(config=True)

    t = T()
    reveal_type(
        Unicode(  # R: traitlets.traitlets.Unicode[builtins.str, Union[builtins.str, builtins.bytes]]
            "foo"
        )
    )
    reveal_type(
        Unicode(  # R: traitlets.traitlets.Unicode[builtins.str, Union[builtins.str, builtins.bytes]]
            ""
        ).tag(sync=True)
    )
    reveal_type(
        Unicode(  # R: traitlets.traitlets.Unicode[Union[builtins.str, None], Union[builtins.str, builtins.bytes, None]]
            None, allow_none=True
        )
    )
    reveal_type(
        Unicode(  # R: traitlets.traitlets.Unicode[Union[builtins.str, None], Union[builtins.str, builtins.bytes, None]]
            None, allow_none=True
        ).tag(sync=True)
    )
    reveal_type(
        T.export_format  # R: traitlets.traitlets.Unicode[builtins.str, Union[builtins.str, builtins.bytes]]
    )
    reveal_type(t.export_format)  # R: builtins.str


@pytest.mark.mypy_testing
def mypy_enum_typing() -> None:
    class T(HasTraits):
        log_level = Enum(
            (0, 10, 20, 30, 40, 50),
            default_value=logging.WARN,
            help="Set the log level by value or name.",
        ).tag(config=True)

    t = T()
    reveal_type(
        Enum(  # R: traitlets.traitlets.Enum[builtins.str]
            ("foo",)
        )
    )
    reveal_type(
        Enum(  # R: traitlets.traitlets.Enum[builtins.str]
            [""]
        ).tag(sync=True)
    )
    reveal_type(
        Enum(  # R: traitlets.traitlets.Enum[None]
            None, allow_none=True
        )
    )
    reveal_type(
        Enum(  # R: traitlets.traitlets.Enum[None]
            None, allow_none=True
        ).tag(sync=True)
    )
    reveal_type(
        T.log_level  # R: traitlets.traitlets.Enum[builtins.int]
    )
    reveal_type(t.log_level)  # R: builtins.int


@pytest.mark.mypy_testing
def mypy_set_typing() -> None:
    class T(HasTraits):
        remove_cell_tags = Set(
            Unicode(),
            default_value=[],
            help=(
                "Tags indicating which cells are to be removed,"
                "matches tags in ``cell.metadata.tags``."
            ),
        ).tag(config=True)

        safe_output_keys = Set(
            config=True,
            default_value={
                "metadata",  # Not a mimetype per-se, but expected and safe.
                "text/plain",
                "text/latex",
                "application/json",
                "image/png",
                "image/jpeg",
            },
            help="Cell output mimetypes to render without modification",
        )

    t = T()
    reveal_type(Set("foo"))  # R: traitlets.traitlets.Set
    reveal_type(Set("").tag(sync=True))  # R: traitlets.traitlets.Set
    reveal_type(Set(None, allow_none=True))  # R: traitlets.traitlets.Set
    reveal_type(Set(None, allow_none=True).tag(sync=True))  # R: traitlets.traitlets.Set
    reveal_type(T.remove_cell_tags)  # R: traitlets.traitlets.Set
    reveal_type(t.remove_cell_tags)  # R: builtins.set[Any]
    reveal_type(T.safe_output_keys)  # R: traitlets.traitlets.Set
    reveal_type(t.safe_output_keys)  # R: builtins.set[Any]


@pytest.mark.mypy_testing
def mypy_any_typing() -> None:
    class T(HasTraits):
        attributes = Any(
            config=True,
            default_value={
                "a": ["href", "title"],
                "abbr": ["title"],
                "acronym": ["title"],
            },
            help="Allowed HTML tag attributes",
        )

    t = T()
    reveal_type(Any("foo"))  # R: traitlets.traitlets.Any
    reveal_type(Any("").tag(sync=True))  # R: traitlets.traitlets.Any
    reveal_type(Any(None, allow_none=True))  # R: traitlets.traitlets.Any
    reveal_type(Any(None, allow_none=True).tag(sync=True))  # R: traitlets.traitlets.Any
    reveal_type(T.attributes)  # R: traitlets.traitlets.Any
    reveal_type(t.attributes)  # R: Any


@pytest.mark.mypy_testing
def mypy_bool_typing() -> None:
    class T(HasTraits):
        b = Bool(True).tag(sync=True)
        ob = Bool(None, allow_none=True).tag(sync=True)

    t = T()
    reveal_type(
        Bool(True)  # R: traitlets.traitlets.Bool[builtins.bool, Union[builtins.bool, builtins.int]]
    )
    reveal_type(
        Bool(  # R: traitlets.traitlets.Bool[builtins.bool, Union[builtins.bool, builtins.int]]
            True
        ).tag(sync=True)
    )
    reveal_type(
        Bool(  # R: traitlets.traitlets.Bool[Union[builtins.bool, None], Union[builtins.bool, builtins.int, None]]
            None, allow_none=True
        )
    )
    reveal_type(
        Bool(  # R: traitlets.traitlets.Bool[Union[builtins.bool, None], Union[builtins.bool, builtins.int, None]]
            None, allow_none=True
        ).tag(sync=True)
    )
    reveal_type(
        T.b  # R: traitlets.traitlets.Bool[builtins.bool, Union[builtins.bool, builtins.int]]
    )
    reveal_type(t.b)  # R: builtins.bool
    reveal_type(t.ob)  # R: Union[builtins.bool, None]
    reveal_type(
        T.b  # R: traitlets.traitlets.Bool[builtins.bool, Union[builtins.bool, builtins.int]]
    )
    reveal_type(
        T.ob  # R: traitlets.traitlets.Bool[Union[builtins.bool, None], Union[builtins.bool, builtins.int, None]]
    )
    # we would expect this to be Optional[Union[bool, int]], but...
    t.b = "foo"  # E: Incompatible types in assignment (expression has type "str", variable has type "Union[bool, int]")  [assignment]
    t.b = None  # E: Incompatible types in assignment (expression has type "None", variable has type "Union[bool, int]")  [assignment]


@pytest.mark.mypy_testing
def mypy_int_typing() -> None:
    class T(HasTraits):
        i: Int[int, int] = Int(42).tag(sync=True)
        oi: Int[int | None, int | None] = Int(42, allow_none=True).tag(sync=True)

    t = T()
    reveal_type(Int(True))  # R: traitlets.traitlets.Int[builtins.int, builtins.int]
    reveal_type(Int(True).tag(sync=True))  # R: traitlets.traitlets.Int[builtins.int, builtins.int]
    reveal_type(
        Int(  # R: traitlets.traitlets.Int[Union[builtins.int, None], Union[builtins.int, None]]
            None, allow_none=True
        )
    )
    reveal_type(
        Int(  # R: traitlets.traitlets.Int[Union[builtins.int, None], Union[builtins.int, None]]
            None, allow_none=True
        ).tag(sync=True)
    )
    reveal_type(T.i)  # R: traitlets.traitlets.Int[builtins.int, builtins.int]
    reveal_type(t.i)  # R: builtins.int
    reveal_type(t.oi)  # R: Union[builtins.int, None]
    reveal_type(T.i)  # R: traitlets.traitlets.Int[builtins.int, builtins.int]
    reveal_type(
        T.oi  # R: traitlets.traitlets.Int[Union[builtins.int, None], Union[builtins.int, None]]
    )
    t.i = "foo"  # E: Incompatible types in assignment (expression has type "str", variable has type "int")  [assignment]
    t.i = None  # E: Incompatible types in assignment (expression has type "None", variable has type "int")  [assignment]
    t.i = 1.2  # E: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]


@pytest.mark.mypy_testing
def mypy_cint_typing() -> None:
    class T(HasTraits):
        i = CInt(42).tag(sync=True)
        oi = CInt(42, allow_none=True).tag(sync=True)

    t = T()
    reveal_type(CInt(42))  # R: traitlets.traitlets.CInt[builtins.int, Any]
    reveal_type(CInt(42).tag(sync=True))  # R: traitlets.traitlets.CInt[builtins.int, Any]
    reveal_type(
        CInt(None, allow_none=True)  # R: traitlets.traitlets.CInt[Union[builtins.int, None], Any]
    )
    reveal_type(
        CInt(  # R: traitlets.traitlets.CInt[Union[builtins.int, None], Any]
            None, allow_none=True
        ).tag(sync=True)
    )
    reveal_type(T.i)  # R: traitlets.traitlets.CInt[builtins.int, Any]
    reveal_type(t.i)  # R: builtins.int
    reveal_type(t.oi)  # R: Union[builtins.int, None]
    reveal_type(T.i)  # R: traitlets.traitlets.CInt[builtins.int, Any]
    reveal_type(T.oi)  # R: traitlets.traitlets.CInt[Union[builtins.int, None], Any]


@pytest.mark.mypy_testing
def mypy_tcp_typing() -> None:
    class T(HasTraits):
        tcp = TCPAddress()
        otcp = TCPAddress(None, allow_none=True)

    t = T()
    reveal_type(t.tcp)  # R: Tuple[builtins.str, builtins.int]
    reveal_type(
        T.tcp  # R: traitlets.traitlets.TCPAddress[Tuple[builtins.str, builtins.int], Tuple[builtins.str, builtins.int]]
    )
    reveal_type(
        T.tcp.tag(  # R:traitlets.traitlets.TCPAddress[Tuple[builtins.str, builtins.int], Tuple[builtins.str, builtins.int]]
            sync=True
        )
    )
    reveal_type(t.otcp)  # R: Union[Tuple[builtins.str, builtins.int], None]
    reveal_type(
        T.otcp  # R: traitlets.traitlets.TCPAddress[Union[Tuple[builtins.str, builtins.int], None], Union[Tuple[builtins.str, builtins.int], None]]
    )
    reveal_type(
        T.otcp.tag(  # R: traitlets.traitlets.TCPAddress[Union[Tuple[builtins.str, builtins.int], None], Union[Tuple[builtins.str, builtins.int], None]]
            sync=True
        )
    )
    t.tcp = "foo"  # E: Incompatible types in assignment (expression has type "str", variable has type "Tuple[str, int]")  [assignment]
    t.otcp = "foo"  # E: Incompatible types in assignment (expression has type "str", variable has type "Optional[Tuple[str, int]]")  [assignment]
    t.tcp = None  # E: Incompatible types in assignment (expression has type "None", variable has type "Tuple[str, int]")  [assignment]


@pytest.mark.mypy_testing
def mypy_instance_typing() -> None:
    class T(HasTraits):
        inst = Instance(Foo)
        oinst = Instance(Foo, allow_none=True)
        oinst_string = Instance("Foo", allow_none=True)

    t = T()
    reveal_type(t.inst)  # R: tests.test_typing.Foo
    reveal_type(T.inst)  # R: traitlets.traitlets.Instance[tests.test_typing.Foo]
    reveal_type(T.inst.tag(sync=True))  # R: traitlets.traitlets.Instance[tests.test_typing.Foo]
    reveal_type(t.oinst)  # R: Union[tests.test_typing.Foo, None]
    reveal_type(t.oinst_string)  # R: Union[Any, None]
    reveal_type(T.oinst)  # R: traitlets.traitlets.Instance[Union[tests.test_typing.Foo, None]]
    reveal_type(
        T.oinst.tag(  # R: traitlets.traitlets.Instance[Union[tests.test_typing.Foo, None]]
            sync=True
        )
    )
    t.inst = "foo"  # E: Incompatible types in assignment (expression has type "str", variable has type "Foo")  [assignment]
    t.oinst = "foo"  # E: Incompatible types in assignment (expression has type "str", variable has type "Optional[Foo]")  [assignment]
    t.inst = None  # E: Incompatible types in assignment (expression has type "None", variable has type "Foo")  [assignment]
