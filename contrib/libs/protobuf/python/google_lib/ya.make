LIBRARY()



NO_COMPILER_WARNINGS()

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
    protobuf/source_context_pb2.py
    protobuf/field_mask_pb2.py
    protobuf/wrappers_pb2.py
    protobuf/compiler/plugin_pb2.py
    protobuf/compiler/__init__.py
    protobuf/text_format.py
    protobuf/descriptor_pb2.py
    protobuf/pyext/cpp_message.py
    protobuf/pyext/__init__.py
    protobuf/pyext/python.proto
    protobuf/proto_builder.py
    protobuf/text_encoding.py
    protobuf/symbol_database.py
    protobuf/__init__.py
    protobuf/reflection.py
    protobuf/any_pb2.py
    protobuf/service_reflection.py
    protobuf/descriptor_pool.py
    protobuf/api_pb2.py
    protobuf/duration_pb2.py
    protobuf/timestamp_pb2.py
    protobuf/internal/wire_format.py
    protobuf/internal/python_message.py
    protobuf/internal/api_implementation.py
    protobuf/internal/message_listener.py
    protobuf/internal/__init__.py
    protobuf/internal/well_known_types.py
    protobuf/internal/encoder.py
    protobuf/internal/enum_type_wrapper.py
    protobuf/internal/_parameterized.py
    protobuf/internal/decoder.py
    protobuf/internal/containers.py
    protobuf/internal/type_checkers.py
    protobuf/service.py
    protobuf/empty_pb2.py
    protobuf/message.py
    protobuf/json_format.py
    protobuf/message_factory.py
    protobuf/type_pb2.py
    protobuf/struct_pb2.py
    protobuf/descriptor_database.py
    protobuf/descriptor.py
)

SRCS(
    protobuf/internal/api_implementation.cc
    protobuf/pyext/repeated_scalar_container.cc
    protobuf/pyext/repeated_composite_container.cc
    protobuf/pyext/descriptor_database.cc
    protobuf/pyext/descriptor_containers.cc
    protobuf/pyext/descriptor.cc
    protobuf/pyext/map_container.cc
    protobuf/pyext/message.cc
    protobuf/pyext/descriptor_pool.cc
    protobuf/pyext/extension_dict.cc
    protobuf/pyext/message_module.cc
)

PY_REGISTER(google.protobuf.pyext._message=_message)
PY_REGISTER(google.protobuf.internal._api_implementation=_api_implementation)

END()
