LIBRARY()



SET(
    RAGEL6_FLAGS
    -L
    -G2
)

SRCS(
    abbreviations.cpp
    charclasses.cpp
    multitokenparser.cpp
    multitokenutil.cpp
    nlpparser.cpp
    sentbreakfilter.cpp
    split.cpp
    tokenizer.cpp
    nlptok_v2.rl6
    nlptok_v3.rl6
    special_tokens.cpp
)

PEERDIR(
    library/enumbitset
    library/langmask
    library/cpp/token
)

IF(CATBOOST_OPENSOURCE)
    CFLAGS(-DCATBOOST_OPENSOURCE=yes)
ELSE()
    PEERDIR(
        library/charset
    )
ENDIF()

RUN_PROGRAM(
    tools/triecompiler special_tokens.trie -0 -i special_tokens.txt -w
    CWD ${ARCADIA_ROOT}/library/cpp/tokenizer
    OUT_NOAUTO special_tokens.trie
    IN special_tokens.txt
)

ARCHIVE_ASM(
    NAME SpecialTokens
    DONTCOMPRESS
    special_tokens.trie
)

END()
