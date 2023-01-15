

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/tools/enum_parser/enum_parser/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    INCLUDE(${ARCADIA_ROOT}/tools/enum_parser/enum_parser/bin/ya.make)
ENDIF()

RECURSE(
    bin
)
