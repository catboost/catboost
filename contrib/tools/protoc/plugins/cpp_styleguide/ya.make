

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/cpp_styleguide/ya.make.prebuilt)
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

    END()
ENDIF()
