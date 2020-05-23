

UNITTEST()

PEERDIR(
    ADDINCL library/cpp/token
    library/cpp/tokenizer
    kernel/reqerror
    kernel/qtree/request
)

SRCDIR(library/cpp/token)

SRCS(
    char_normalization_ut.cpp
    charfilter_ut.cpp
    token_iterator_ut.cpp
)

END()
