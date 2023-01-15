

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/rescompressor/ya.make)
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
