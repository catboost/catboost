

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/protoc/ya.make.prebuilt)
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

    END()
ENDIF()
