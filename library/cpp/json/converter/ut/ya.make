

UNITTEST_FOR(library/cpp/json/converter)

SIZE(SMALL)

SRCS(
    test_conversion.cpp
)

PEERDIR(
    library/cpp/json
    library/cpp/json/writer
)

END()
