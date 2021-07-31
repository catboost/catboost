#!/usr/bin/env python
"""Protoc Plugin to generate mypy stubs. Loosely based on @zbarsky's go implementation"""
from __future__ import (
    absolute_import,
    division,
    print_function,
)

import sys
from collections import defaultdict
from contextlib import contextmanager

import google.protobuf.descriptor_pb2 as d
import six
from google.protobuf.compiler import plugin_pb2 as plugin_pb2

MYPY = False
if MYPY:
    from typing import (
        Any,
        Callable,
        Dict,
        Generator,
        Iterable,
        List,
        Set,
        Sequence,
        Text,
        Tuple,
        cast,
    )
    from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
else:
    # Provide minimal mypy identifiers to make code run without typing module present
    Text = None

    def cast(type, value):
        return value

GENERATED = "@ge" + "nerated"  # So phabricator doesn't think this file is generated
HEADER = "# {} by generate_proto_mypy_stubs.py.  Do not edit!\n".format(GENERATED)

# See https://github.com/dropbox/mypy-protobuf/issues/73 for details
PYTHON_RESERVED = {
    'False',
    'None',
    'True',
    'and',
    'as',
    'async',
    'await',
    'assert',
    'break',
    'class',
    'continue',
    'def',
    'del',
    'elif',
    'else',
    'except',
    'finally',
    'for',
    'from',
    'global',
    'if',
    'import',
    'in',
    'is',
    'lambda',
    'nonlocal',
    'not',
    'or',
    'pass',
    'raise',
    'return',
    'try',
    'while',
    'with',
    'yield',
}



FORWARD_REFERENCE_STRING_LITERAL = True


def forward_ref(name):
    if FORWARD_REFERENCE_STRING_LITERAL:
        return "'{}'".format(name)
    else:
        return name


class PkgWriter(object):
    """Writes a single pyi file"""

    def __init__(self, fd, descriptors):
        # type: (d.FileDescriptorProto, Descriptors) -> None
        self.fd = fd
        self.descriptors = descriptors
        self.lines = []  # type: List[Text]
        self.indent = ""

        # dictionary of x->(y,z) for `from {x} import {y} as {z}`
        self.imports = defaultdict(set)  # type: Dict[Text, Set[Tuple[Text, Text]]]
        self.locals = set()  # type: Set[Text]
        self.builtin_vars = set()  # type: Set[Text]

    def _import(self, path, name):
        # type: (Text, Text) -> Text
        """Imports a stdlib path and returns a handle to it
        eg. self._import("typing", "Optional") -> "Optional"
        """
        imp = path.replace('/', '.')
        mangled_name = imp.replace('.', '___') + '___' + name
        self.imports[imp].add((name, mangled_name))
        return mangled_name

    def _import_message(self, name):
        # type: (Text) -> Text
        """Import a referenced message and return a handle"""
        message_fd = self.descriptors.message_to_fd[name]
        assert message_fd.name.endswith('.proto')

        # Strip off package name
        if message_fd.package:
            assert name.startswith("." + message_fd.package + '.')
            name = name[len('.' + message_fd.package + '.'):]
        else:
            assert name.startswith(".")
            name = name[1:]

        # Message defined in this file.
        if message_fd.name == self.fd.name:
            return name

        # Not in file. Must import
        # Python generated code ignores proto packages, so the only relevant factor is
        # whether it is in the file or not.
        split = name.split(".")
        import_name = self._import(message_fd.name[:-6].replace('-', '_') + "_pb2", split[0])
        remains = ".".join(split[1:])
        if not remains:
            return import_name
        # remains could either be a direct import of a nested enum or message
        # from another package.
        return import_name + "." + remains

    def _builtin(self, name):
        self.builtin_vars.add(name)
        return self._mangle_builtin(name)

    @staticmethod
    def _mangle_builtin(name):
        return 'builtin___{}'.format(name)

    @contextmanager
    def _indent(self):
        # type: () -> Generator
        self.indent = self.indent + "    "
        yield
        self.indent = self.indent[:-4]

    def _write_line(self, line, *args):
        # type: (Text, *Any) -> None
        if line == "":
            self.lines.append(line)
        else:
            self.lines.append(self.indent + line.format(*args))

    def write_enum_values(self, enum, enum_full_name):
        for val in enum.value:
            if val.name in PYTHON_RESERVED:
                continue

            self._write_line("{} = {}({}, {})", val.name, self._import("typing", "cast"), enum_full_name, val.number)

    def write_enums(self, enums, prefix=''):
        # type: (Iterable[d.EnumDescriptorProto], Text) -> None
        l = self._write_line
        for enum in [e for e in enums if e.name not in PYTHON_RESERVED]:
            # MyPy does not support referencing the enum by the unqualified name *inside* nested enums
            enum_full_name = forward_ref(prefix + enum.name)

            l("class {}({}):", enum.name, self._builtin('int'))
            with self._indent():
                l("DESCRIPTOR: {} = ...",
                    self._import("google.protobuf.descriptor", "EnumDescriptor"))
                l("@classmethod")
                l("def Name(cls, number: {}) -> {}: ...", self._builtin('int'), self._builtin('str'))
                l("@classmethod")
                l("def Value(cls, name: {}) -> {}: ...", self._builtin('str'), enum_full_name)
                l("@classmethod")
                l("def keys(cls) -> {}[{}]: ...",
                    self._import("typing", "List"), self._builtin('str'))
                l("@classmethod")
                l("def values(cls) -> {}[{}]: ...",
                    self._import("typing", "List"),
                    enum_full_name)
                l("@classmethod")
                l("def items(cls) -> {}[{}[{}, {}]]: ...",
                    self._import("typing", "List"),
                    self._import("typing", "Tuple"),
                    self._builtin('str'),
                    enum_full_name)

                self.write_enum_values(enum, enum_full_name)

            self.write_enum_values(enum, enum_full_name)
            l("")

    def write_messages(self, messages, prefix):
        # type: (Iterable[d.DescriptorProto], Text) -> None
        l = self._write_line
        message_class = self._import("google.protobuf.message", "Message")

        for desc in [m for m in messages if m.name not in PYTHON_RESERVED]:
            self.locals.add(desc.name)
            qualified_name = prefix + desc.name
            l("class {}({}):", desc.name, message_class)
            with self._indent():
                l("DESCRIPTOR: {} = ...",
                    self._import("google.protobuf.descriptor", "Descriptor"))

                # Nested enums/messages
                self.write_enums(desc.enum_type, qualified_name + ".")
                self.write_messages(desc.nested_type, qualified_name + ".")
                fields = [f for f in desc.field if f.name not in PYTHON_RESERVED]

                # Scalar fields
                for field in [f for f in fields if is_scalar(f)]:
                    if field.label == d.FieldDescriptorProto.LABEL_REPEATED:
                        container = self._import("google.protobuf.internal.containers", "RepeatedScalarFieldContainer")
                        l("{} = ... # type: {}[{}]", field.name, container, self.python_type(field))
                    else:
                        l("{} = ... # type: {}", field.name, self.python_type(field))
                l("")

                # Getters for non-scalar fields
                for field in [f for f in fields if not is_scalar(f)]:
                    l("@property")
                    if field.label == d.FieldDescriptorProto.LABEL_REPEATED:
                        msg = self.descriptors.messages[field.type_name]
                        if msg.options.map_entry:
                            # map generates a special Entry wrapper message
                            container = self._import("typing", "MutableMapping")
                            l("def {}(self) -> {}[{}, {}]: ...", field.name, container, self.python_type(msg.field[0]), self.python_type(msg.field[1]))
                        else:
                            container = self._import("google.protobuf.internal.containers", "RepeatedCompositeFieldContainer")
                            l("def {}(self) -> {}[{}]: ...", field.name, container, self.python_type(field))
                    else:
                        l("def {}(self) -> {}: ...", field.name, self.python_type(field))
                    l("")

                # Constructor
                l("def __init__(self,")
                with self._indent():
                    if len(fields) > 0:
                        # Only positional args allowed
                        # See https://github.com/dropbox/mypy-protobuf/issues/71
                        l("*,")
                    # Required args
                    for field in [f for f in fields if f.label == d.FieldDescriptorProto.LABEL_REQUIRED]:
                        l("{} : {},", field.name, self.python_type(field))
                    for field in [f for f in fields if f.label != d.FieldDescriptorProto.LABEL_REQUIRED]:
                        if field.label == d.FieldDescriptorProto.LABEL_REPEATED:
                            if field.type_name in self.descriptors.messages and self.descriptors.messages[field.type_name].options.map_entry:
                                msg = self.descriptors.messages[field.type_name]
                                l("{} : {}[{}[{}, {}]] = None,", field.name, self._import("typing", "Optional"),
                                    self._import("typing", "Mapping"), self.python_type(msg.field[0]), self.python_type(msg.field[1]))
                            else:
                                l("{} : {}[{}[{}]] = None,", field.name, self._import("typing", "Optional"),
                                  self._import("typing", "Iterable"), self.python_type(field))
                        else:
                            l("{} : {}[{}] = None,", field.name, self._import("typing", "Optional"),
                              self.python_type(field))
                    l(") -> None: ...")

                # Standard message methods
                l("@classmethod")
                l("def FromString(cls, s: {}) -> {}: ...", self._builtin('bytes'), qualified_name)
                l("def MergeFrom(self, other_msg: {}) -> None: ...", message_class)
                l("def CopyFrom(self, other_msg: {}) -> None: ...", message_class)

                self.write_stringly_typed_fields(desc)

            l("")

    def write_stringly_typed_fields(self, desc):
        # type: (d.DescriptorProto) -> None
        """Type the stringly-typed methods as a Union[Literal, Literal ...]"""
        l = self._write_line
        # HasField accepts bytes/unicode in PY2, but only unicode in PY3
        # ClearField accepts bytes/unicode in PY2 and unicode in PY3
        # WhichOneof accepts bytes/unicode in both PY2 and PY3
        #
        # HasField only supports singular. ClearField supports repeated as well
        #
        # In proto3, HasField only supports message fields
        #
        # HasField always supports oneof fields
        hf_fields = [
            f.name for f in desc.field
            if f.HasField("oneof_index") or
                (f.label != d.FieldDescriptorProto.LABEL_REPEATED and
                    (self.fd.syntax != "proto3" or f.type == d.FieldDescriptorProto.TYPE_MESSAGE))
        ]
        cf_fields = [f.name for f in desc.field]
        wo_fields = {
            oneof.name: [f.name for f in desc.field if f.HasField("oneof_index") and f.oneof_index == idx]
            for idx, oneof in enumerate(desc.oneof_decl)
        }

        hf_fields.extend(wo_fields.keys())
        cf_fields.extend(wo_fields.keys())

        hf_fields_py3 = ",".join(sorted('u"{}"'.format(name) for name in hf_fields))
        cf_fields_py3 = ",".join(sorted('u"{}"'.format(name) for name in cf_fields))
        hf_fields_py2 = ",".join(sorted('u"{}",b"{}"'.format(name, name) for name in hf_fields))
        cf_fields_py2 = ",".join(sorted('u"{}",b"{}"'.format(name, name) for name in cf_fields))

        if not hf_fields and not cf_fields and not wo_fields:
            return

        l("if {} >= (3,):", "sys.version_info")
        with self._indent():
            if hf_fields:
                l("def HasField(self, field_name: {}[{}]) -> {}: ...",
                    self._import("typing_extensions", "Literal"),
                    hf_fields_py3,
                    self._builtin('bool'))
            if cf_fields:
                l("def ClearField(self, field_name: {}[{}]) -> None: ...",
                    self._import("typing_extensions", "Literal"),
                    cf_fields_py3)
        l("else:")
        with self._indent():
            if hf_fields:
                l("def HasField(self, field_name: {}[{}]) -> {}: ...",
                    self._import("typing_extensions", "Literal"),
                    hf_fields_py2,
                    self._builtin('bool'))
            if cf_fields:
                l("def ClearField(self, field_name: {}[{}]) -> None: ...",
                    self._import("typing_extensions", "Literal"),
                    cf_fields_py2)
        for wo_field, members in sorted(wo_fields.items()):
            if len(wo_fields) > 1:
                l("@{}", self._import("typing", "overload"))
            l("def WhichOneof(self, oneof_group: {}[{}]) -> {}[{}]: ...",
                self._import("typing_extensions", "Literal"),
                # Accepts both unicode and bytes in both py2 and py3
                'u"{}",b"{}"'.format(wo_field, wo_field),
                self._import("typing_extensions", "Literal"),
                # Returns `str` in both py2 and py3 (bytes in py2, unicode in py3)
                ",".join('"{}"'.format(m) for m in members))

    def write_extensions(self, extensions):
        # type: (Sequence[d.FieldDescriptorProto]) -> None
        if not extensions:
            return
        l = self._write_line
        field_descriptor_class = self._import("google.protobuf.descriptor", "FieldDescriptor")
        for extension in extensions:
            l("{} = ... # type: {}", extension.name, field_descriptor_class)
            l("")

    def write_methods(self, service, is_abstract):
        # type: (d.ServiceDescriptorProto, bool) -> None
        l = self._write_line
        methods = [m for m in service.method if m.name not in PYTHON_RESERVED]
        if not methods:
            l("pass")
        for method in methods:
            if is_abstract:
                l("@{}", self._import("abc", "abstractmethod"))
            l("def {}(self,", method.name)
            with self._indent():
                l("rpc_controller: {},", self._import("google.protobuf.service", "RpcController"))
                l("request: {},", self._import_message(method.input_type))
                l("done: {}[{}[[{}], None]],",
                  self._import("typing", "Optional"),
                  self._import("typing", "Callable"),
                  self._import_message(method.output_type))
            l(") -> {}[{}]: ...", self._import("concurrent.futures", "Future"), self._import_message(method.output_type))

    def write_services(self, services):
        # type: (Iterable[d.ServiceDescriptorProto]) -> None
        l = self._write_line
        for service in [s for s in services if s.name not in PYTHON_RESERVED]:
            # The service definition interface
            l("class {}({}, metaclass={}):", service.name, self._import("google.protobuf.service", "Service"), self._import("abc", "ABCMeta"))
            with self._indent():
                self.write_methods(service, is_abstract=True)

            # The stub client
            l("class {}({}):", service.name + "_Stub", service.name)
            with self._indent():
                l("def __init__(self, rpc_channel: {}) -> None: ...",
                  self._import("google.protobuf.service", "RpcChannel"))
                self.write_methods(service, is_abstract=False)

    def python_type(self, field):
        # type: (d.FieldDescriptorProto) -> Text
        b_float = self._builtin('float')
        b_int = self._builtin('int')
        b_bool = self._builtin('bool')
        b_bytes = self._builtin('bytes')

        mapping = {
            d.FieldDescriptorProto.TYPE_DOUBLE: lambda: b_float,
            d.FieldDescriptorProto.TYPE_FLOAT: lambda: b_float,

            d.FieldDescriptorProto.TYPE_INT64: lambda: b_int,
            d.FieldDescriptorProto.TYPE_UINT64: lambda: b_int,
            d.FieldDescriptorProto.TYPE_FIXED64: lambda: b_int,
            d.FieldDescriptorProto.TYPE_SFIXED64: lambda: b_int,
            d.FieldDescriptorProto.TYPE_SINT64: lambda: b_int,
            d.FieldDescriptorProto.TYPE_INT32: lambda: b_int,
            d.FieldDescriptorProto.TYPE_UINT32: lambda: b_int,
            d.FieldDescriptorProto.TYPE_FIXED32: lambda: b_int,
            d.FieldDescriptorProto.TYPE_SFIXED32: lambda: b_int,
            d.FieldDescriptorProto.TYPE_SINT32: lambda: b_int,

            d.FieldDescriptorProto.TYPE_BOOL: lambda: b_bool,
            d.FieldDescriptorProto.TYPE_STRING: lambda: self._import("typing", "Text"),
            d.FieldDescriptorProto.TYPE_BYTES: lambda: b_bytes,

            d.FieldDescriptorProto.TYPE_ENUM: lambda: self._import_message(field.type_name),
            d.FieldDescriptorProto.TYPE_MESSAGE: lambda: self._import_message(field.type_name),
            d.FieldDescriptorProto.TYPE_GROUP: lambda: self._import_message(field.type_name),
        }  # type: Dict[d.FieldDescriptorProto.Type, Callable[[], Text]]

        assert field.type in mapping, "Unrecognized type: " + repr(field.type)
        return mapping[field.type]()

    def write(self):
        # type: () -> Text
        imports = []
        imports.append(u"import sys")
        for pkg, items in sorted(six.iteritems(self.imports)):
            imports.append(u"from {} import (".format(pkg))
            for (name, mangled_name) in sorted(items):
                imports.append(u"    {} as {},".format(name, mangled_name))
            imports.append(u")\n")
        imports.append("")

        aliases = []
        for var in sorted(self.builtin_vars):
            aliases.append(u'{} = {}'.format(self._mangle_builtin(var), var))
        if aliases:
            aliases.append('\n')

        return "\n".join(imports + aliases + self.lines)


def is_scalar(fd):
    # type: (d.FieldDescriptorProto) -> bool
    return not (
        fd.type == d.FieldDescriptorProto.TYPE_MESSAGE or
        fd.type == d.FieldDescriptorProto.TYPE_GROUP
    )


def generate_mypy_stubs(descriptors, response, quiet):
    # type: (Descriptors, plugin_pb2.CodeGeneratorResponse, bool) -> None
    for name, fd in six.iteritems(descriptors.to_generate):
        pkg_writer = PkgWriter(fd, descriptors)
        pkg_writer.write_enums(fd.enum_type)
        pkg_writer.write_messages(fd.message_type, "")
        pkg_writer.write_extensions(fd.extension)
        if fd.options.py_generic_services:
            pkg_writer.write_services(fd.service)

        assert name == fd.name
        assert fd.name.endswith('.proto')
        output = response.file.add()
        output.name = fd.name[:-6].replace('-', '_').replace('.', '/') + '_pb2.pyi'
        output.content = HEADER + pkg_writer.write()


class Descriptors(object):

    def __init__(self, request):
        # type: (plugin_pb2.CodeGeneratorRequest) -> None
        files = {f.name: f for f in request.proto_file}
        to_generate = {n: files[n] for n in request.file_to_generate}
        self.files = files  # type: Dict[Text, d.FileDescriptorProto]
        self.to_generate = to_generate  # type: Dict[Text, d.FileDescriptorProto]
        self.messages = {} # type: Dict[Text, d.DescriptorProto]
        self.message_to_fd = {}  # type: Dict[Text, d.FileDescriptorProto]

        def _add_enums(enums, prefix, _fd):
            # type: (RepeatedCompositeFieldContainer[d.EnumDescriptorProto], Text, d.FileDescriptorProto) -> None
            for enum in enums:
                self.message_to_fd[prefix + enum.name] = _fd

        def _add_messages(messages, prefix, _fd):
            # type: (RepeatedCompositeFieldContainer[d.DescriptorProto], Text, d.FileDescriptorProto) -> None
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


def main():
    # type: () -> None
    # Read request message from stdin
    if six.PY3:
        data = sys.stdin.buffer.read()
    else:
        data = sys.stdin.read()

    # Parse request
    request = plugin_pb2.CodeGeneratorRequest()
    request.ParseFromString(data)

    # Create response
    response = plugin_pb2.CodeGeneratorResponse()

    # Generate mypy
    generate_mypy_stubs(Descriptors(request), response, "quiet" in request.parameter)

    # Serialise response message
    output = response.SerializeToString()

    # Write to stdout
    if six.PY3:
        sys.stdout.buffer.write(output)
    else:
        sys.stdout.write(output)


if __name__ == '__main__':
    main()
