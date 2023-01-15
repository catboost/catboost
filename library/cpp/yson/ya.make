LIBRARY()



SRCS(
    yson2json_adapter.cpp
    consumer.cpp
    json_writer.cpp
    lexer.cpp
    parser.cpp
    token.cpp
    tokenizer.cpp
    varint.cpp
    writer.cpp
    zigzag.h
)

PEERDIR(
    util
    library/cpp/json
)

END()
