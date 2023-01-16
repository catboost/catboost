#!/usr/bin/env python
"""Protoc Plugin to generate mypy stubs. Loosely based on @zbarsky's go implementation"""
import os

import sys
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
)

import google.protobuf.descriptor_pb2 as d
from google.protobuf.compiler import plugin_pb2 as plugin_pb2
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from google.protobuf.internal.well_known_types import WKTBASES
from . import extensions_pb2

__version__ = "2.10"

# SourceCodeLocation is defined by `message Location` here
# https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/descriptor.proto
SourceCodeLocation = List[int]

# So phabricator doesn't think mypy_protobuf.py is generated
GENERATED = "@ge" + "nerated"
HEADER = """\"\"\"
{} by mypy-protobuf.  Do not edit manually!
isort:skip_file
\"\"\"
""".format(
    GENERATED
)

# See https://github.com/dropbox/mypy-protobuf/issues/73 for details
PYTHON_RESERVED = {
    "False",
    "None",
    "True",
    "and",
    "as",
    "async",
    "await",
    "assert",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}

PROTO_ENUM_RESERVED = {
    "Name",
    "Value",
    "keys",
    "values",
    "items",
}


def _mangle_global_identifier(name: str) -> str:
    """
    Module level identifiers are mangled and aliased so that they can be disambiguated
    from fields/enum variants with the same name within the file.

    Eg:
    Enum variant `Name` or message field `Name` might conflict with a top level
    message or enum named `Name`, so mangle it with a global___ prefix for
    internal references. Note that this doesn't affect inner enums/messages
    because they get fuly qualified when referenced within a file"""
    return "global___{}".format(name)


class Descriptors(object):
    def __init__(self, request: plugin_pb2.CodeGeneratorRequest) -> None:
        files = {f.name: f for f in request.proto_file}
        to_generate = {n: files[n] for n in request.file_to_generate}
        self.files: Dict[str, d.FileDescriptorProto] = files
        self.to_generate: Dict[str, d.FileDescriptorProto] = to_generate
        self.messages: Dict[str, d.DescriptorProto] = {}
        self.message_to_fd: Dict[str, d.FileDescriptorProto] = {}

        def _add_enums(
            enums: "RepeatedCompositeFieldContainer[d.EnumDescriptorProto]",
            prefix: str,
            _fd: d.FileDescriptorProto,
        ) -> None:
            for enum in enums:
                self.message_to_fd[prefix + enum.name] = _fd
                self.message_to_fd[prefix + enum.name + ".V"] = _fd

        def _add_messages(
            messages: "RepeatedCompositeFieldContainer[d.DescriptorProto]",
            prefix: str,
            _fd: d.FileDescriptorProto,
        ) -> None:
            for message in messages:
                self.messages[prefix + message.name] = message
                self.message_to_fd[prefix + message.name] = _fd
                sub_prefix = prefix + message.name + "."
                _add_messages(message.nested_type, sub_prefix, _fd)
                _add_enums(message.enum_type, sub_prefix, _fd)

        for fd in request.proto_file:
            start_prefix = "." + fd.package + "." if fd.package else "."
            _add_messages(fd.message_type, start_prefix, fd)
            _add_enums(fd.enum_type, start_prefix, fd)


class PkgWriter(object):
    """Writes a single pyi file"""

    def __init__(
        self,
        fd: d.FileDescriptorProto,
        descriptors: Descriptors,
        readable_stubs: bool,
        relax_strict_optional_primitives: bool,
        grpc: bool,
    ) -> None:
        self.fd = fd
        self.descriptors = descriptors
        self.readable_stubs = readable_stubs
        self.relax_strict_optional_primitives = relax_strict_optional_primitives
        self.grpc = grpc
        self.lines: List[str] = []
        self.indent = ""

        # Set of {x}, where {x} corresponds to to `import {x}`
        self.imports: Set[str] = set()
        # dictionary of x->(y,z) for `from {x} import {y} as {z}`
        # if {z} is None, then it shortens to `from {x} import {y}`
        self.from_imports: Dict[str, Set[Tuple[str, Optional[str]]]] = defaultdict(set)

        # Comments
        self.source_code_info_by_scl = {
            tuple(location.path): location for location in fd.source_code_info.location
        }

    def _import(self, path: str, name: str) -> str:
        """Imports a stdlib path and returns a handle to it
        eg. self._import("typing", "Optional") -> "Optional"
        """
        imp = path.replace("/", ".")
        if self.readable_stubs:
            self.from_imports[imp].add((name, None))
            return name
        else:
            self.imports.add(imp)
            return imp + "." + name

    def _import_message(self, name: str) -> str:
        """Import a referenced message and return a handle"""
        message_fd = self.descriptors.message_to_fd[name]
        assert message_fd.name.endswith(".proto")

        # Strip off package name
        if message_fd.package:
            assert name.startswith("." + message_fd.package + ".")
            name = name[len("." + message_fd.package + ".") :]
        else:
            assert name.startswith(".")
            name = name[1:]

        # Use prepended "_r_" to disambiguate message names that alias python reserved keywords
        split = name.split(".")
        for i, part in enumerate(split):
            if part in PYTHON_RESERVED:
                split[i] = "_r_" + part
        name = ".".join(split)

        # Message defined in this file. Note: GRPC stubs in same .proto are generated into separate files
        if not self.grpc and message_fd.name == self.fd.name:
            return name if self.readable_stubs else _mangle_global_identifier(name)

        # Not in file. Must import
        # Python generated code ignores proto packages, so the only relevant factor is
        # whether it is in the file or not.
        import_name = self._import(
            message_fd.name[:-6].replace("-", "_") + "_pb2", split[0]
        )

        remains = ".".join(split[1:])
        if not remains:
            return import_name

        # remains could either be a direct import of a nested enum or message
        # from another package.
        return import_name + "." + remains

    def _builtin(self, name: str) -> str:
        return self._import("builtins", name)

    @contextmanager
    def _indent(self) -> Iterator[None]:
        self.indent = self.indent + "    "
        yield
        self.indent = self.indent[:-4]

    def _write_line(self, line: str, *args: Any) -> None:
        line = line.format(*args)
        if line == "":
            self.lines.append(line)
        else:
            self.lines.append(self.indent + line)

    def _break_text(self, text_block: str) -> List[str]:
        if text_block == "":
            return []
        return [
            "{}".format(l[1:] if l.startswith(" ") else l)
            for l in text_block.rstrip().split("\n")
        ]

    def _has_comments(self, scl: SourceCodeLocation) -> bool:
        sci_loc = self.source_code_info_by_scl.get(tuple(scl))
        return sci_loc is not None and bool(
            sci_loc.leading_detached_comments
            or sci_loc.leading_comments
            or sci_loc.trailing_comments
        )

    def _write_comments(self, scl: SourceCodeLocation) -> bool:
        """Return true if any comments were written"""
        if not self._has_comments(scl):
            return False

        sci_loc = self.source_code_info_by_scl.get(tuple(scl))
        assert sci_loc is not None

        lines = []
        for leading_detached_comment in sci_loc.leading_detached_comments:
            lines.extend(self._break_text(leading_detached_comment))
            lines.append("")
        if sci_loc.leading_comments is not None:
            lines.extend(self._break_text(sci_loc.leading_comments))
        # Trailing comments also go in the header - to make sure it gets into the docstring
        if sci_loc.trailing_comments is not None:
            lines.extend(self._break_text(sci_loc.trailing_comments))

        if len(lines) == 1:
            self._write_line('"""{}"""', lines[0])
        else:
            for i, line in enumerate(lines):
                if i == 0:
                    self._write_line('"""{}', line)
                else:
                    self._write_line("{}", line)
            self._write_line('"""')

        return True

    def write_enum_values(
        self,
        values: Iterable[Tuple[int, d.EnumValueDescriptorProto]],
        value_type: str,
        scl_prefix: SourceCodeLocation,
    ) -> None:
        for i, val in values:
            if val.name in PYTHON_RESERVED:
                continue

            scl = scl_prefix + [i]
            self._write_line(
                "{} = {}({})",
                val.name,
                value_type,
                val.number,
            )
            if self._write_comments(scl):
                self._write_line("")  # Extra newline to separate

    def write_module_attributes(self) -> None:
        l = self._write_line
        l(
            "DESCRIPTOR: {} = ...",
            self._import("google.protobuf.descriptor", "FileDescriptor"),
        )
        l("")

    def write_enums(
        self,
        enums: Iterable[d.EnumDescriptorProto],
        prefix: str,
        scl_prefix: SourceCodeLocation,
    ) -> None:
        l = self._write_line
        for i, enum in enumerate(enums):
            class_name = (
                enum.name if enum.name not in PYTHON_RESERVED else "_r_" + enum.name
            )
            value_type_fq = prefix + class_name + ".V"

            l(
                "class {}({}, metaclass={}):",
                class_name,
                "_" + enum.name,
                "_" + enum.name + "EnumTypeWrapper",
            )
            with self._indent():
                scl = scl_prefix + [i]
                self._write_comments(scl)
                l("pass")
            l("class {}:", "_" + enum.name)
            with self._indent():
                l(
                    "V = {}('V', {})",
                    self._import("typing", "NewType"),
                    self._builtin("int"),
                )
            l(
                "class {}({}[{}], {}):",
                "_" + enum.name + "EnumTypeWrapper",
                self._import(
                    "google.protobuf.internal.enum_type_wrapper", "_EnumTypeWrapper"
                ),
                "_" + enum.name + ".V",
                self._builtin("type"),
            )
            with self._indent():
                l(
                    "DESCRIPTOR: {} = ...",
                    self._import("google.protobuf.descriptor", "EnumDescriptor"),
                )
                self.write_enum_values(
                    [
                        (i, v)
                        for i, v in enumerate(enum.value)
                        if v.name not in PROTO_ENUM_RESERVED
                    ],
                    value_type_fq,
                    scl + [d.EnumDescriptorProto.VALUE_FIELD_NUMBER],
                )
            l("")

            self.write_enum_values(
                enumerate(enum.value),
                value_type_fq,
                scl + [d.EnumDescriptorProto.VALUE_FIELD_NUMBER],
            )
            if prefix == "" and not self.readable_stubs:
                l("{} = {}", _mangle_global_identifier(class_name), class_name)
                l("")
            l("")

    def write_messages(
        self,
        messages: Iterable[d.DescriptorProto],
        prefix: str,
        scl_prefix: SourceCodeLocation,
    ) -> None:
        l = self._write_line

        for i, desc in enumerate(messages):
            qualified_name = prefix + desc.name

            # Reproduce some hardcoded logic from the protobuf implementation - where
            # some specific "well_known_types" generated protos to have additional
            # base classes
            addl_base = u""
            if self.fd.package + "." + desc.name in WKTBASES:
                # chop off the .proto - and import the well known type
                # eg `from google.protobuf.duration import Duration`
                well_known_type = WKTBASES[self.fd.package + "." + desc.name]
                addl_base = ", " + self._import(
                    "google.protobuf.internal.well_known_types",
                    well_known_type.__name__,
                )

            class_name = (
                desc.name if desc.name not in PYTHON_RESERVED else "_r_" + desc.name
            )
            message_class = self._import("google.protobuf.message", "Message")
            l("class {}({}{}):", class_name, message_class, addl_base)
            with self._indent():
                scl = scl_prefix + [i]
                self._write_comments(scl)

                l(
                    "DESCRIPTOR: {} = ...",
                    self._import("google.protobuf.descriptor", "Descriptor"),
                )

                # Nested enums/messages
                self.write_enums(
                    desc.enum_type,
                    qualified_name + ".",
                    scl + [d.DescriptorProto.ENUM_TYPE_FIELD_NUMBER],
                )
                self.write_messages(
                    desc.nested_type,
                    qualified_name + ".",
                    scl + [d.DescriptorProto.NESTED_TYPE_FIELD_NUMBER],
                )

                # integer constants  for field numbers
                for f in desc.field:
                    l("{}_FIELD_NUMBER: {}", f.name.upper(), self._builtin("int"))

                for idx, field in enumerate(desc.field):
                    if field.name in PYTHON_RESERVED:
                        continue

                    if (
                        is_scalar(field)
                        and field.label != d.FieldDescriptorProto.LABEL_REPEATED
                    ):
                        # Scalar non repeated fields are r/w
                        l("{}: {} = ...", field.name, self.python_type(field))
                        if self._write_comments(
                            scl + [d.DescriptorProto.FIELD_FIELD_NUMBER, idx]
                        ):
                            l("")
                    else:
                        # r/o Getters for non-scalar fields and scalar-repeated fields
                        scl_field = scl + [d.DescriptorProto.FIELD_FIELD_NUMBER, idx]
                        l("@property")
                        l(
                            "def {}(self) -> {}:{}",
                            field.name,
                            self.python_type(field),
                            " ..." if not self._has_comments(scl_field) else "",
                        )
                        if self._has_comments(scl_field):
                            with self._indent():
                                self._write_comments(scl_field)
                                l("pass")

                self.write_extensions(
                    desc.extension, scl + [d.DescriptorProto.EXTENSION_FIELD_NUMBER]
                )

                # Constructor
                self_arg = (
                    "self_" if any(f.name == "self" for f in desc.field) else "self"
                )
                l("def __init__({},", self_arg)
                with self._indent():
                    constructor_fields = [
                        f for f in desc.field if f.name not in PYTHON_RESERVED
                    ]
                    if len(constructor_fields) > 0:
                        # Only positional args allowed
                        # See https://github.com/dropbox/mypy-protobuf/issues/71
                        l("*,")
                    for field in constructor_fields:
                        if (
                            self.fd.syntax == "proto3"
                            and is_scalar(field)
                            and field.label != d.FieldDescriptorProto.LABEL_REPEATED
                            and not self.relax_strict_optional_primitives
                        ):
                            l(
                                "{} : {} = ...,",
                                field.name,
                                self.python_type(field, generic_container=True),
                            )
                        else:
                            l(
                                "{} : {}[{}] = ...,",
                                field.name,
                                self._import("typing", "Optional"),
                                self.python_type(field, generic_container=True),
                            )
                    l(") -> None: ...")

                self.write_stringly_typed_fields(desc)

            if prefix == "" and not self.readable_stubs:
                l("{} = {}", _mangle_global_identifier(class_name), class_name)
            l("")

    def write_stringly_typed_fields(self, desc: d.DescriptorProto) -> None:
        """Type the stringly-typed methods as a Union[Literal, Literal ...]"""
        l = self._write_line
        # HasField, ClearField, WhichOneof accepts both bytes/unicode
        # HasField only supports singular. ClearField supports repeated as well
        # In proto3, HasField only supports message fields and optional fields
        # HasField always supports oneof fields
        hf_fields = [
            f.name
            for f in desc.field
            if f.HasField("oneof_index")
            or (
                f.label != d.FieldDescriptorProto.LABEL_REPEATED
                and (
                    self.fd.syntax != "proto3"
                    or f.type == d.FieldDescriptorProto.TYPE_MESSAGE
                    or f.proto3_optional
                )
            )
        ]
        cf_fields = [f.name for f in desc.field]
        wo_fields = {
            oneof.name: [
                f.name
                for f in desc.field
                if f.HasField("oneof_index") and f.oneof_index == idx
            ]
            for idx, oneof in enumerate(desc.oneof_decl)
        }

        hf_fields.extend(wo_fields.keys())
        cf_fields.extend(wo_fields.keys())

        hf_fields_text = ",".join(
            sorted('u"{}",b"{}"'.format(name, name) for name in hf_fields)
        )
        cf_fields_text = ",".join(
            sorted('u"{}",b"{}"'.format(name, name) for name in cf_fields)
        )

        if not hf_fields and not cf_fields and not wo_fields:
            return

        if hf_fields:
            l(
                "def HasField(self, field_name: {}[{}]) -> {}: ...",
                self._import("typing_extensions", "Literal"),
                hf_fields_text,
                self._builtin("bool"),
            )
        if cf_fields:
            l(
                "def ClearField(self, field_name: {}[{}]) -> None: ...",
                self._import("typing_extensions", "Literal"),
                cf_fields_text,
            )

        for wo_field, members in sorted(wo_fields.items()):
            if len(wo_fields) > 1:
                l("@{}", self._import("typing", "overload"))
            l(
                "def WhichOneof(self, oneof_group: {}[{}]) -> {}[{}[{}]]: ...",
                self._import("typing_extensions", "Literal"),
                # Accepts both unicode and bytes in both py2 and py3
                'u"{}",b"{}"'.format(wo_field, wo_field),
                self._import("typing", "Optional"),
                self._import("typing_extensions", "Literal"),
                # Returns `str` in both py2 and py3 (bytes in py2, unicode in py3)
                ",".join('"{}"'.format(m) for m in members),
            )

    def write_extensions(
        self,
        extensions: Sequence[d.FieldDescriptorProto],
        scl_prefix: SourceCodeLocation,
    ) -> None:
        l = self._write_line
        for i, ext in enumerate(extensions):
            scl = scl_prefix + [i]

            l(
                "{}: {}[{}, {}] = ...",
                ext.name,
                self._import(
                    "google.protobuf.internal.extension_dict",
                    "_ExtensionFieldDescriptor",
                ),
                self._import_message(ext.extendee),
                self.python_type(ext),
            )
            self._write_comments(scl)
            l("")

    def write_methods(
        self,
        service: d.ServiceDescriptorProto,
        is_abstract: bool,
        scl_prefix: SourceCodeLocation,
    ) -> None:
        l = self._write_line
        methods = [
            (i, m)
            for i, m in enumerate(service.method)
            if m.name not in PYTHON_RESERVED
        ]
        if not methods:
            l("pass")
        for i, method in methods:
            if is_abstract:
                l("@{}", self._import("abc", "abstractmethod"))
            l("def {}(self,", method.name)
            with self._indent():
                l(
                    "rpc_controller: {},",
                    self._import("google.protobuf.service", "RpcController"),
                )
                l("request: {},", self._import_message(method.input_type))
                l(
                    "done: {}[{}[[{}], None]],",
                    self._import("typing", "Optional"),
                    self._import("typing", "Callable"),
                    self._import_message(method.output_type),
                )

            scl_method = scl_prefix + [d.ServiceDescriptorProto.METHOD_FIELD_NUMBER, i]
            l(
                ") -> {}[{}]:{}",
                self._import("concurrent.futures", "Future"),
                self._import_message(method.output_type),
                " ..." if not self._has_comments(scl_method) else "",
            )
            if self._has_comments(scl_method):
                with self._indent():
                    self._write_comments(scl_method)
                    l("pass")

    def write_services(
        self,
        services: Iterable[d.ServiceDescriptorProto],
        scl_prefix: SourceCodeLocation,
    ) -> None:
        l = self._write_line
        for i, service in enumerate(services):
            scl = scl_prefix + [i]
            class_name = (
                service.name
                if service.name not in PYTHON_RESERVED
                else "_r_" + service.name
            )
            # The service definition interface
            l(
                "class {}({}, metaclass={}):",
                class_name,
                self._import("google.protobuf.service", "Service"),
                self._import("abc", "ABCMeta"),
            )
            with self._indent():
                self._write_comments(scl)
                self.write_methods(service, is_abstract=True, scl_prefix=scl)

            # The stub client
            l("class {}({}):", service.name + "_Stub", class_name)
            with self._indent():
                self._write_comments(scl)
                l(
                    "def __init__(self, rpc_channel: {}) -> None: ...",
                    self._import("google.protobuf.service", "RpcChannel"),
                )
                self.write_methods(service, is_abstract=False, scl_prefix=scl)

    def _import_casttype(self, casttype: str) -> str:
        split = casttype.split(".")
        assert (
            len(split) == 2
        ), "mypy_protobuf.[casttype,keytype,valuetype] is expected to be of format path/to/file.TypeInFile"
        pkg = split[0].replace("/", ".")
        return self._import(pkg, split[1])

    def _map_key_value_types(
        self,
        map_field: d.FieldDescriptorProto,
        key_field: d.FieldDescriptorProto,
        value_field: d.FieldDescriptorProto,
    ) -> Tuple[str, str]:
        key_casttype = map_field.options.Extensions[extensions_pb2.keytype]
        ktype = (
            self._import_casttype(key_casttype)
            if key_casttype
            else self.python_type(key_field)
        )
        value_casttype = map_field.options.Extensions[extensions_pb2.valuetype]
        vtype = (
            self._import_casttype(value_casttype)
            if value_casttype
            else self.python_type(value_field)
        )
        return ktype, vtype

    def _callable_type(self, method: d.MethodDescriptorProto) -> str:
        if method.client_streaming:
            if method.server_streaming:
                return self._import("grpc", "StreamStreamMultiCallable")
            else:
                return self._import("grpc", "StreamUnaryMultiCallable")
        else:
            if method.server_streaming:
                return self._import("grpc", "UnaryStreamMultiCallable")
            else:
                return self._import("grpc", "UnaryUnaryMultiCallable")

    def _input_type(
        self, method: d.MethodDescriptorProto, use_stream_iterator: bool = True
    ) -> str:
        result = self._import_message(method.input_type)
        if use_stream_iterator and method.client_streaming:
            result = "{}[{}]".format(self._import("typing", "Iterator"), result)
        return result

    def _output_type(
        self, method: d.MethodDescriptorProto, use_stream_iterator: bool = True
    ) -> str:
        result = self._import_message(method.output_type)
        if use_stream_iterator and method.server_streaming:
            result = "{}[{}]".format(self._import("typing", "Iterator"), result)
        return result

    def write_grpc_methods(
        self, service: d.ServiceDescriptorProto, scl_prefix: SourceCodeLocation
    ) -> None:
        l = self._write_line
        methods = [
            (i, m)
            for i, m in enumerate(service.method)
            if m.name not in PYTHON_RESERVED
        ]
        if not methods:
            l("pass")
            l("")
        for i, method in methods:
            scl = scl_prefix + [d.ServiceDescriptorProto.METHOD_FIELD_NUMBER, i]

            l("@{}", self._import("abc", "abstractmethod"))
            l("def {}(self,", method.name)
            with self._indent():
                l("request: {},", self._input_type(method))
                l("context: {},", self._import("grpc", "ServicerContext"))
            l(
                ") -> {}:{}",
                self._output_type(method),
                " ..." if not self._has_comments(scl) else "",
            ),
            if self._has_comments(scl):
                with self._indent():
                    self._write_comments(scl)
                    l("pass")
            l("")

    def write_grpc_stub_methods(
        self, service: d.ServiceDescriptorProto, scl_prefix: SourceCodeLocation
    ) -> None:
        l = self._write_line
        methods = [
            (i, m)
            for i, m in enumerate(service.method)
            if m.name not in PYTHON_RESERVED
        ]
        if not methods:
            l("pass")
            l("")
        for i, method in methods:
            scl = scl_prefix + [d.ServiceDescriptorProto.METHOD_FIELD_NUMBER, i]

            l("{}: {}[", method.name, self._callable_type(method))
            with self._indent():
                l("{},", self._input_type(method, False))
                l("{}] = ...", self._output_type(method, False))
            self._write_comments(scl)
            l("")

    def write_grpc_services(
        self,
        services: Iterable[d.ServiceDescriptorProto],
        scl_prefix: SourceCodeLocation,
    ) -> None:
        l = self._write_line
        for i, service in enumerate(services):
            if service.name in PYTHON_RESERVED:
                continue

            scl = scl_prefix + [i]

            # The stub client
            l("class {}Stub:", service.name)
            with self._indent():
                self._write_comments(scl)
                l(
                    "def __init__(self, channel: {}) -> None: ...",
                    self._import("grpc", "Channel"),
                )
                self.write_grpc_stub_methods(service, scl)
            l("")

            # The service definition interface
            l(
                "class {}Servicer(metaclass={}):",
                service.name,
                self._import("abc", "ABCMeta"),
            )
            with self._indent():
                self._write_comments(scl)
                self.write_grpc_methods(service, scl)
            l("")
            l(
                "def add_{}Servicer_to_server(servicer: {}Servicer, server: {}) -> None: ...",
                service.name,
                service.name,
                self._import("grpc", "Server"),
            )
            l("")

    def python_type(
        self, field: d.FieldDescriptorProto, generic_container: bool = False
    ) -> str:
        """
        generic_container
          if set, type the field with generic interfaces. Eg.
          - Iterable[int] rather than RepeatedScalarFieldContainer[int]
          - Mapping[k, v] rather than MessageMap[k, v]
          Can be useful for input types (eg constructor)
        """
        casttype = field.options.Extensions[extensions_pb2.casttype]
        if casttype:
            return self._import_casttype(casttype)

        mapping: Dict[d.FieldDescriptorProto.Type.V, Callable[[], str]] = {
            d.FieldDescriptorProto.TYPE_DOUBLE: lambda: self._builtin("float"),
            d.FieldDescriptorProto.TYPE_FLOAT: lambda: self._builtin("float"),
            d.FieldDescriptorProto.TYPE_INT64: lambda: self._builtin("int"),
            d.FieldDescriptorProto.TYPE_UINT64: lambda: self._builtin("int"),
            d.FieldDescriptorProto.TYPE_FIXED64: lambda: self._builtin("int"),
            d.FieldDescriptorProto.TYPE_SFIXED64: lambda: self._builtin("int"),
            d.FieldDescriptorProto.TYPE_SINT64: lambda: self._builtin("int"),
            d.FieldDescriptorProto.TYPE_INT32: lambda: self._builtin("int"),
            d.FieldDescriptorProto.TYPE_UINT32: lambda: self._builtin("int"),
            d.FieldDescriptorProto.TYPE_FIXED32: lambda: self._builtin("int"),
            d.FieldDescriptorProto.TYPE_SFIXED32: lambda: self._builtin("int"),
            d.FieldDescriptorProto.TYPE_SINT32: lambda: self._builtin("int"),
            d.FieldDescriptorProto.TYPE_BOOL: lambda: self._builtin("bool"),
            d.FieldDescriptorProto.TYPE_STRING: lambda: self._import("typing", "Text"),
            d.FieldDescriptorProto.TYPE_BYTES: lambda: self._builtin("bytes"),
            d.FieldDescriptorProto.TYPE_ENUM: lambda: self._import_message(
                field.type_name + ".V"
            ),
            d.FieldDescriptorProto.TYPE_MESSAGE: lambda: self._import_message(
                field.type_name
            ),
            d.FieldDescriptorProto.TYPE_GROUP: lambda: self._import_message(
                field.type_name
            ),
        }

        assert field.type in mapping, "Unrecognized type: " + repr(field.type)
        field_type = mapping[field.type]()

        # For non-repeated fields, we're done!
        if field.label != d.FieldDescriptorProto.LABEL_REPEATED:
            return field_type

        # Scalar repeated fields go in RepeatedScalarFieldContainer
        if is_scalar(field):
            container = (
                self._import("typing", "Iterable")
                if generic_container
                else self._import(
                    "google.protobuf.internal.containers",
                    "RepeatedScalarFieldContainer",
                )
            )
            return "{}[{}]".format(container, field_type)

        # non-scalar repeated map fields go in ScalarMap/MessageMap
        msg = self.descriptors.messages[field.type_name]
        if msg.options.map_entry:
            # map generates a special Entry wrapper message
            if generic_container:
                container = self._import("typing", "Mapping")
            elif is_scalar(msg.field[1]):
                container = self._import(
                    "google.protobuf.internal.containers", "ScalarMap"
                )
            else:
                container = self._import(
                    "google.protobuf.internal.containers", "MessageMap"
                )
            ktype, vtype = self._map_key_value_types(field, msg.field[0], msg.field[1])
            return "{}[{}, {}]".format(container, ktype, vtype)

        # non-scalar repetated fields go in RepeatedCompositeFieldContainer
        container = (
            self._import("typing", "Iterable")
            if generic_container
            else self._import(
                "google.protobuf.internal.containers",
                "RepeatedCompositeFieldContainer",
            )
        )
        return "{}[{}]".format(container, field_type)

    def write(self) -> str:
        for reexport_idx in self.fd.public_dependency:
            reexport_file = self.fd.dependency[reexport_idx]
            reexport_fd = self.descriptors.files[reexport_file]
            reexport_imp = (
                reexport_file[:-6].replace("-", "_").replace("/", ".") + "_pb2"
            )
            names = (
                [m.name for m in reexport_fd.message_type]
                + [m.name for m in reexport_fd.enum_type]
                + [v.name for m in reexport_fd.enum_type for v in m.value]
                + [m.name for m in reexport_fd.extension]
            )
            if reexport_fd.options.py_generic_services:
                names.extend(m.name for m in reexport_fd.service)

            if names:
                # n,n to force a reexport (from x import y as y)
                self.from_imports[reexport_imp].update((n, n) for n in names)

        import_lines = []
        for pkg in sorted(self.imports):
            import_lines.append(u"import {}".format(pkg))

        for pkg, items in sorted(self.from_imports.items()):
            import_lines.append(u"from {} import (".format(pkg))
            for (name, reexport_name) in sorted(items):
                if reexport_name is None:
                    import_lines.append(u"    {},".format(name))
                else:
                    import_lines.append(u"    {} as {},".format(name, reexport_name))
            import_lines.append(u")\n")
        import_lines.append("")

        return "\n".join(import_lines + self.lines)


def is_scalar(fd: d.FieldDescriptorProto) -> bool:
    return not (
        fd.type == d.FieldDescriptorProto.TYPE_MESSAGE
        or fd.type == d.FieldDescriptorProto.TYPE_GROUP
    )


def generate_mypy_stubs(
    descriptors: Descriptors,
    response: plugin_pb2.CodeGeneratorResponse,
    quiet: bool,
    readable_stubs: bool,
    relax_strict_optional_primitives: bool,
) -> None:
    for name, fd in descriptors.to_generate.items():
        pkg_writer = PkgWriter(
            fd,
            descriptors,
            readable_stubs,
            relax_strict_optional_primitives,
            grpc=False,
        )

        pkg_writer.write_module_attributes()
        pkg_writer.write_enums(
            fd.enum_type, "", [d.FileDescriptorProto.ENUM_TYPE_FIELD_NUMBER]
        )
        pkg_writer.write_messages(
            fd.message_type, "", [d.FileDescriptorProto.MESSAGE_TYPE_FIELD_NUMBER]
        )
        pkg_writer.write_extensions(
            fd.extension, [d.FileDescriptorProto.EXTENSION_FIELD_NUMBER]
        )
        if fd.options.py_generic_services:
            pkg_writer.write_services(
                fd.service, [d.FileDescriptorProto.SERVICE_FIELD_NUMBER]
            )

        assert name == fd.name
        assert fd.name.endswith(".proto")
        output = response.file.add()
        output.name = fd.name[:-6].replace("-", "_").replace(".", "/") + "_pb2.pyi"
        output.content = HEADER + pkg_writer.write()


def generate_mypy_grpc_stubs(
    descriptors: Descriptors,
    response: plugin_pb2.CodeGeneratorResponse,
    quiet: bool,
    readable_stubs: bool,
    relax_strict_optional_primitives: bool,
) -> None:
    for name, fd in descriptors.to_generate.items():
        pkg_writer = PkgWriter(
            fd,
            descriptors,
            readable_stubs,
            relax_strict_optional_primitives,
            grpc=True,
        )
        pkg_writer.write_grpc_services(
            fd.service, [d.FileDescriptorProto.SERVICE_FIELD_NUMBER]
        )

        assert name == fd.name
        assert fd.name.endswith(".proto")
        output = response.file.add()
        output.name = fd.name[:-6].replace("-", "_").replace(".", "/") + "_pb2_grpc.pyi"
        output.content = HEADER + pkg_writer.write()


@contextmanager
def code_generation() -> Iterator[
    Tuple[plugin_pb2.CodeGeneratorRequest, plugin_pb2.CodeGeneratorResponse],
]:
    if len(sys.argv) > 1 and sys.argv[1] in ("-V", "--version"):
        print("mypy-protobuf " + __version__)
        sys.exit(0)

    # Read request message from stdin
    data = sys.stdin.buffer.read()

    # Parse request
    request = plugin_pb2.CodeGeneratorRequest()
    request.ParseFromString(data)

    # Create response
    response = plugin_pb2.CodeGeneratorResponse()

    # Declare support for optional proto3 fields
    response.supported_features |= (
        plugin_pb2.CodeGeneratorResponse.FEATURE_PROTO3_OPTIONAL
    )

    yield request, response

    # Serialise response message
    output = response.SerializeToString()

    # Write to stdout
    sys.stdout.buffer.write(output)


def main() -> None:
    # Generate mypy
    with code_generation() as (request, response):
        generate_mypy_stubs(
            Descriptors(request),
            response,
            "quiet" in request.parameter,
            "readable_stubs" in request.parameter,
            "relax_strict_optional_primitives" in request.parameter,
        )


def grpc() -> None:
    # Generate grpc mypy
    with code_generation() as (request, response):
        generate_mypy_grpc_stubs(
            Descriptors(request),
            response,
            "quiet" in request.parameter,
            "readable_stubs" in request.parameter,
            "relax_strict_optional_primitives" in request.parameter,
        )


if __name__ == "__main__":
    main()
