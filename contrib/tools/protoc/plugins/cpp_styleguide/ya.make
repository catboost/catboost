

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/protoc/plugins/cpp_styleguide/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    INCLUDE(${ARCADIA_ROOT}/contrib/tools/protoc/plugins/cpp_styleguide/bin/ya.make)
ENDIF()

RECURSE(
    bin
)
