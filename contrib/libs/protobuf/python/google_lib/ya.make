PY23_LIBRARY()
LICENSE(
    BSD3
)



NO_CHECK_IMPORTS(
    google.protobuf.internal.cpp_message
    google.protobuf.pyext.cpp_message
)

NO_COMPILER_WARNINGS()

NO_LINT()

OPTIMIZE_PY_PROTOS()

PEERDIR(
    ADDINCL contrib/libs/protobuf
    contrib/python/six
)

ADDINCL(
    contrib/libs/protobuf/python
    contrib/libs/protobuf/python/google/protobuf
)

SRCDIR(
    contrib/libs/protobuf/python/google
)

CFLAGS(-DPYTHON_PROTO2_CPP_IMPL_V2)

PY_SRCS(
    NAMESPACE google
    __init__.py
    protobuf/__init__.py
    protobuf/any_pb2.py
    protobuf/api_pb2.py
    protobuf/compiler/__init__.py
    protobuf/compiler/plugin_pb2.py
    protobuf/descriptor.py
    protobuf/descriptor_database.py
    protobuf/descriptor_pb2.py
    protobuf/descriptor_pool.py
    protobuf/duration_pb2.py
    protobuf/empty_pb2.py
    protobuf/field_mask_pb2.py
    protobuf/internal/__init__.py
    protobuf/internal/_parameterized.py
    protobuf/internal/api_implementation.py
    protobuf/internal/containers.py
    protobuf/internal/decoder.py
    protobuf/internal/encoder.py
    protobuf/internal/enum_type_wrapper.py
    protobuf/internal/message_listener.py
    protobuf/internal/python_message.py
    protobuf/internal/type_checkers.py
    protobuf/internal/well_known_types.py
    protobuf/internal/wire_format.py
    protobuf/json_format.py
    protobuf/message.py
    protobuf/message_factory.py
    protobuf/proto_builder.py
    protobuf/pyext/__init__.py
    protobuf/pyext/cpp_message.py
    protobuf/pyext/python.proto
    protobuf/reflection.py
    protobuf/service.py
    protobuf/service_reflection.py
    protobuf/source_context_pb2.py
    protobuf/struct_pb2.py
    protobuf/symbol_database.py
    protobuf/text_encoding.py
    protobuf/text_format.py
    protobuf/timestamp_pb2.py
    protobuf/type_pb2.py
    protobuf/wrappers_pb2.py
)

SRCS(
    protobuf/internal/api_implementation.cc
    protobuf/internal/python_protobuf.cc
    protobuf/pyext/descriptor.cc
    protobuf/pyext/descriptor_containers.cc
    protobuf/pyext/descriptor_database.cc
    protobuf/pyext/descriptor_pool.cc
    protobuf/pyext/extension_dict.cc
    protobuf/pyext/map_container.cc
    protobuf/pyext/message.cc
    protobuf/pyext/message_factory.cc
    protobuf/pyext/message_module.cc
    protobuf/pyext/repeated_composite_container.cc
    protobuf/pyext/repeated_scalar_container.cc
)

PY_REGISTER(google.protobuf.pyext._message)
PY_REGISTER(google.protobuf.internal._api_implementation)

END()
