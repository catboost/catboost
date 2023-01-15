UNITTEST()



PEERDIR(
    ADDINCL tools/enum_parser/parse_enum
    library/resource
)

SRCDIR(tools/enum_parser/parse_enum)

RESOURCE(
    enums.h /enums
    badcode.h /badcode
    unbalanced.h /unbalanced
    alias_before_name.h /alias_before_name
)

# self-test
GENERATE_ENUM_SERIALIZATION(enums.h)

# test GENERATE_ENUM_SERIALIZATION_WITH_HEADER macro
GENERATE_ENUM_SERIALIZATION_WITH_HEADER(enums_with_header.h)

SRCS(
    parse_enum_ut.cpp
    enums.cpp
)

END()
