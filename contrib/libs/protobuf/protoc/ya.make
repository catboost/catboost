LIBRARY()

LICENSE(
    BSD3
)



JOINSRC()

NO_COMPILER_WARNINGS()

PEERDIR(
    ADDINCL contrib/libs/protobuf
)

SRCDIR(
    contrib/libs/protobuf
    contrib/libs/protobuf/compiler
    contrib/libs/protobuf/compiler/cpp
    contrib/libs/protobuf/compiler/java
    contrib/libs/protobuf/compiler/js
    contrib/libs/protobuf/compiler/perlxs
    contrib/libs/protobuf/compiler/python
    contrib/libs/protobuf/io
    contrib/libs/protobuf/stubs
)

SRCS(
    code_generator.cc
    command_line_interface.cc
    cpp_enum.cc
    cpp_enum_field.cc
    cpp_extension.cc
    cpp_field.cc
    cpp_file.cc
    cpp_generator.cc
    cpp_helpers.cc
    cpp_map_field.cc
    cpp_message.cc
    cpp_message_field.cc
    cpp_primitive_field.cc
    cpp_service.cc
    cpp_string_field.cc
    java_context.cc
    java_doc_comment.cc
    java_enum.cc
    java_enum_field.cc
    java_enum_field_lite.cc
    java_enum_lite.cc
    java_extension.cc
    java_extension_lite.cc
    java_field.cc
    java_file.cc
    java_generator.cc
    java_generator_factory.cc
    java_helpers.cc
    java_lazy_message_field.cc
    java_lazy_message_field_lite.cc
    java_map_field.cc
    java_map_field_lite.cc
    java_message.cc
    java_message_builder.cc
    java_message_builder_lite.cc
    java_message_field.cc
    java_message_field_lite.cc
    java_message_lite.cc
    java_name_resolver.cc
    java_primitive_field.cc
    java_primitive_field_lite.cc
    java_service.cc
    java_shared_code_generator.cc
    java_string_field.cc
    java_string_field_lite.cc
    js_generator.cc
    perlxs_generator.cc
    perlxs_helpers.cc
    plugin.cc
    plugin.pb.cc
    python_generator.cc
    subprocess.cc
    well_known_types_embed.cpp
    zip_writer.cc
)

SET(IDE_FOLDER "contrib/libs")

END()
