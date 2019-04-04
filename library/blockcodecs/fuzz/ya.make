

IF (NOT MSVC)
    FUZZ()

    SIZE(MEDIUM)

    SRCS(
        main.cpp
    )

    PEERDIR(
        contrib/libs/protobuf
        contrib/libs/protobuf-mutator
        library/blockcodecs
        library/blockcodecs/fuzz/proto
    )

    END()
ENDIF()
