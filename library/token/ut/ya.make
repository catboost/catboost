

UNITTEST()

PEERDIR(
    ADDINCL library/token
    library/tokenizer
    kernel/reqerror
    kernel/qtree/request
)

SRCDIR(library/token)

SRCS(
    char_normalization_ut.cpp
    charfilter_ut.cpp
    token_iterator_ut.cpp
)

END()
