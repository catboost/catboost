

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/tools/rorescompiler/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    PROGRAM()

    PEERDIR(
        library/cpp/resource
    )

    SRCS(
        main.cpp
    )

    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/tools/rorescompiler/ya.make.induced_deps)

    END()
ENDIF()
