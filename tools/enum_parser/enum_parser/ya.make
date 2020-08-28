

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/tools/enum_parser/enum_parser/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    PROGRAM()

    SRCS(
        main.cpp
    )

    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/tools/enum_parser/enum_parser/ya.make.induced_deps)

    PEERDIR(
        library/cpp/getopt/small
        tools/enum_parser/parse_enum
    )

    END()
ENDIF()
