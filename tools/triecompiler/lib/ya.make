

LIBRARY()

SRCS(
    main.cpp
)

PEERDIR(
    library/cpp/containers/comptrie
    library/cpp/deprecated/mapped_file
    library/cpp/getopt/small
)

IF(NOT CATBOOST_OPENSOURCE)
    PEERDIR(
        library/charset
    )
ENDIF()

END()
