

LIBRARY()

PROVIDES(protoc)

LICENSE(
    BSD3
)

NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/libs/protobuf
)
ADDINCL(
    GLOBAL contrib/libs/protoc/src
    # Temporary here, MUST be removed upon re-syncing includes with upstream
    GLOBAL contrib/libs/protoc/src/google/protobuf
)

SRCS(
    src/google/protobuf/compiler/code_generator.cc
    src/google/protobuf/compiler/command_line_interface.cc
    src/google/protobuf/compiler/cpp/cpp_enum.cc
    src/google/protobuf/compiler/cpp/cpp_enum_field.cc
    src/google/protobuf/compiler/cpp/cpp_extension.cc
    src/google/protobuf/compiler/cpp/cpp_field.cc
    src/google/protobuf/compiler/cpp/cpp_file.cc
    src/google/protobuf/compiler/cpp/cpp_generator.cc
    src/google/protobuf/compiler/cpp/cpp_helpers.cc
    src/google/protobuf/compiler/cpp/cpp_map_field.cc
    src/google/protobuf/compiler/cpp/cpp_message.cc
    src/google/protobuf/compiler/cpp/cpp_message_field.cc
    src/google/protobuf/compiler/cpp/cpp_primitive_field.cc
    src/google/protobuf/compiler/cpp/cpp_service.cc
    src/google/protobuf/compiler/cpp/cpp_string_field.cc
    src/google/protobuf/compiler/importer.cc
    src/google/protobuf/compiler/java/java_context.cc
    src/google/protobuf/compiler/java/java_doc_comment.cc
    src/google/protobuf/compiler/java/java_enum.cc
    src/google/protobuf/compiler/java/java_enum_field.cc
    src/google/protobuf/compiler/java/java_enum_field_lite.cc
    src/google/protobuf/compiler/java/java_enum_lite.cc
    src/google/protobuf/compiler/java/java_extension.cc
    src/google/protobuf/compiler/java/java_extension_lite.cc
    src/google/protobuf/compiler/java/java_field.cc
    src/google/protobuf/compiler/java/java_file.cc
    src/google/protobuf/compiler/java/java_generator.cc
    src/google/protobuf/compiler/java/java_generator_factory.cc
    src/google/protobuf/compiler/java/java_helpers.cc
    src/google/protobuf/compiler/java/java_lazy_message_field.cc
    src/google/protobuf/compiler/java/java_lazy_message_field_lite.cc
    src/google/protobuf/compiler/java/java_map_field.cc
    src/google/protobuf/compiler/java/java_map_field_lite.cc
    src/google/protobuf/compiler/java/java_message.cc
    src/google/protobuf/compiler/java/java_message_builder.cc
    src/google/protobuf/compiler/java/java_message_builder_lite.cc
    src/google/protobuf/compiler/java/java_message_field.cc
    src/google/protobuf/compiler/java/java_message_field_lite.cc
    src/google/protobuf/compiler/java/java_message_lite.cc
    src/google/protobuf/compiler/java/java_name_resolver.cc
    src/google/protobuf/compiler/java/java_primitive_field.cc
    src/google/protobuf/compiler/java/java_primitive_field_lite.cc
    src/google/protobuf/compiler/java/java_service.cc
    src/google/protobuf/compiler/java/java_shared_code_generator.cc
    src/google/protobuf/compiler/java/java_string_field.cc
    src/google/protobuf/compiler/java/java_string_field_lite.cc
    src/google/protobuf/compiler/js/js_generator.cc
    src/google/protobuf/compiler/js/well_known_types_embed.cpp
    src/google/protobuf/compiler/parser.cc
    src/google/protobuf/compiler/perlxs/perlxs_generator.cc
    src/google/protobuf/compiler/perlxs/perlxs_helpers.cc
    src/google/protobuf/compiler/plugin.cc
    src/google/protobuf/compiler/plugin.pb.cc
    src/google/protobuf/compiler/python/python_generator.cc
    src/google/protobuf/compiler/subprocess.cc
    src/google/protobuf/compiler/zip_writer.cc
)

END()
