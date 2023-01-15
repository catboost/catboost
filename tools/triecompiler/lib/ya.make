

LIBRARY()

SRCS(
    main.cpp
)

PEERDIR(
    library/comptrie
    library/deprecated/mapped_file
    library/cpp/getopt/small
)

IF(NOT CATBOOST_OPENSOURCE)
    PEERDIR(
        library/charset
    )
ENDIF()

END()
