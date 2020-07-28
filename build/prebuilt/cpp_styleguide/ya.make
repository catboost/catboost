

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/cpp_styleguide/ya.make.prebuilt)

IF(NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt cpp_styleguide tool)
ENDIF()
