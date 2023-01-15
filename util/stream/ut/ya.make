UNITTEST_FOR(util)



SRCS(
    stream/aligned_ut.cpp
    stream/buffer_ut.cpp
    stream/buffered_ut.cpp
    stream/direct_io_ut.cpp
    stream/file_ut.cpp
    stream/format_ut.cpp
    stream/hex_ut.cpp
    stream/input_ut.cpp
    stream/ios_ut.cpp
    stream/labeled_ut.cpp
    stream/length_ut.cpp
    stream/mem_ut.cpp
    stream/multi_ut.cpp
    stream/printf_ut.cpp
    stream/str_ut.cpp
    stream/tokenizer_ut.cpp
    stream/walk_ut.cpp
    stream/zerocopy_output_ut.cpp
    stream/zlib_ut.cpp
)

INCLUDE(${ARCADIA_ROOT}/util/tests/ya_util_tests.inc)

END()
