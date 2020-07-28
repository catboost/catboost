

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/uca9dump/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt uca9dump tool)
ENDIF()
