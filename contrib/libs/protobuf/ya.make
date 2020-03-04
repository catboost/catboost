LIBRARY()

LICENSE(
    BSD3
)



NO_COMPILER_WARNINGS()

SUPPRESSIONS(tsan.supp)

ADDINCLSELF()

ADDINCL(
    GLOBAL contrib/libs/protobuf
    GLOBAL contrib/libs/protobuf/google/protobuf
)

JOINSRC()

IF (OS_ANDROID)
    EXTRALIBS(-llog)
ENDIF()

CFLAGS(-DHAVE_ZLIB)

PEERDIR(contrib/libs/zlib)

SRCS(
    any.cc
    any.pb.cc
    api.pb.cc
    arena.cc
    arenastring.cc
    compiler/importer.cc
    compiler/parser.cc
    descriptor.cc
    descriptor.pb.cc
    descriptor_database.cc
    duration.pb.cc
    dynamic_message.cc
    empty.pb.cc
    extension_set.cc
    extension_set_heavy.cc
    field_mask.pb.cc
    generated_message_reflection.cc
    generated_message_table_driven.cc
    generated_message_table_driven_lite.cc
    generated_message_util.cc
    io/coded_stream.cc
    io/gzip_stream.cc
    io/printer.cc
    io/strtod.cc
    io/tokenizer.cc
    io/zero_copy_stream.cc
    io/zero_copy_stream_impl.cc
    io/zero_copy_stream_impl_lite.cc
    json_util.cc
    map_field.cc
    message.cc
    message_lite.cc
    messagext.cc
    messagext_lite.cc
    reflection_ops.cc
    repeated_field.cc
    service.cc
    source_context.pb.cc
    struct.pb.cc
    stubs/atomicops_internals_x86_gcc.cc
    stubs/atomicops_internals_x86_msvc.cc
    stubs/bytestream.cc
    stubs/common.cc
    stubs/int128.cc
    stubs/io_win32.cc
    stubs/mathlimits.cc
    stubs/once.cc
    stubs/status.cc
    stubs/statusor.cc
    stubs/stringpiece.cc
    stubs/stringprintf.cc
    stubs/structurally_valid.cc
    stubs/strutil.cc
    stubs/substitute.cc
    stubs/time.cc
    text_format.cc
    timestamp.pb.cc
    type.pb.cc
    unknown_field_set.cc
    util/delimited_message_util.cc
    util/field_comparator.cc
    util/field_mask_util.cc
    util/internal/datapiece.cc
    util/internal/default_value_objectwriter.cc
    util/internal/error_listener.cc
    util/internal/field_mask_utility.cc
    util/internal/json_escaping.cc
    util/internal/json_objectwriter.cc
    util/internal/json_stream_parser.cc
    util/internal/object_writer.cc
    util/internal/proto_writer.cc
    util/internal/protostream_objectsource.cc
    util/internal/protostream_objectwriter.cc
    util/internal/type_info.cc
    util/internal/utility.cc
    util/json_util.cc
    util/message_differencer.cc
    util/time_util.cc
    util/type_resolver_util.cc
    wire_format.cc
    wire_format_lite.cc
    wrappers.pb.cc
)

FILES(
    google/protobuf/any.proto
    google/protobuf/api.proto
    google/protobuf/descriptor.proto
    google/protobuf/duration.proto
    google/protobuf/empty.proto
    google/protobuf/field_mask.proto
    google/protobuf/source_context.proto
    google/protobuf/struct.proto
    google/protobuf/timestamp.proto
    google/protobuf/type.proto
    google/protobuf/wrappers.proto
)

END()
