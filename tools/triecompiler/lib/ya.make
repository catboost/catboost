

LIBRARY()

SRCS(
    main.cpp
)

PEERDIR(
    library/cpp/containers/comptrie
    library/cpp/deprecated/mapped_file
    library/cpp/getopt/small
)

IF(CATBOOST_OPENSOURCE)
    CFLAGS(-DCATBOOST_OPENSOURCE)
ELSE()
    PEERDIR(
        library/cpp/charset
    )
ENDIF()

END()
