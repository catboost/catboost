LIBRARY()



PEERDIR(
    library/cpp/yt/misc
    library/cpp/yt/yson
)

SRCS(
    consumer.cpp
    lexer.cpp
    parser.cpp
    token.cpp
    tokenizer.cpp
    varint.cpp
    writer.cpp
)

END()
