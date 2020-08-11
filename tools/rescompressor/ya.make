

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/tools/rescompressor/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    PROGRAM()

    PEERDIR(
        library/cpp/resource
    )

    SRCS(
        main.cpp
    )

    END()
ENDIF()
