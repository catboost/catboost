

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/event2cpp/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt event2cpp tool)
ENDIF()
