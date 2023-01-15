

PROGRAM(enum_parser)

SRCDIR(
    tools/enum_parser/enum_parser
)

SRCS(
    main.cpp
)

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/tools/enum_parser/enum_parser/ya.make.induced_deps)

PEERDIR(
    library/cpp/getopt/small
    tools/enum_parser/parse_enum
)

END()
