

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/fix_elf/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    PROGRAM()

    SRCS(
        patch.cpp
    )

    PEERDIR(
        library/cpp/getopt/small
    )

    END()
ENDIF()
