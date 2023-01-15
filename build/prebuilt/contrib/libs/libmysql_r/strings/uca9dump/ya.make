

INCLUDE(ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt uca9dump tool)
ENDIF()
