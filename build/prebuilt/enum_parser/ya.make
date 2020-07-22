

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/enum_parser/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt enum_parser tool)
ENDIF()
