

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/flatc/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    PROGRAM()

    NO_UTIL()

    ADDINCL(
        contrib/libs/flatbuffers/include
    )

    PEERDIR(
        contrib/libs/flatbuffers/flatc
    )

    SRCDIR(
        contrib/libs/flatbuffers/src
    )

    SRCS(
        flatc_main.cpp
    )

    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/flatc/ya.make.induced_deps)

    END()
ENDIF()
