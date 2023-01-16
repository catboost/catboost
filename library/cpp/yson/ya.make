LIBRARY()



PEERDIR(
    library/cpp/yt/misc
    library/cpp/yt/yson_string
)

SRCS(
    consumer.cpp
    lexer.cpp
    parser.cpp
    string.cpp
    token.cpp
    tokenizer.cpp
    varint.cpp
    writer.cpp
)

END()
