

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/protoc/plugins/cpp_styleguide/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    PROGRAM()

    NO_COMPILER_WARNINGS()

    PEERDIR(
        contrib/libs/protoc
    )

    SRCS(
        cpp_styleguide.cpp
    )

    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/protoc/plugins/cpp_styleguide/ya.make.induced_deps)

    END()
ENDIF()
