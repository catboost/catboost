

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/protoc/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    PROGRAM()

    LICENSE(
        BSD3
    )

    NO_COMPILER_WARNINGS()

    PEERDIR(
        contrib/libs/protoc
    )
    SRCDIR(
        contrib/libs/protoc
    )

    SRCS(
        src/google/protobuf/compiler/main.cc
    )

    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/protoc/ya.make.induced_deps)
    END()
ENDIF()
