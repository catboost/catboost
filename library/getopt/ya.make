LIBRARY()



IF (YMAKE)
    CFLAGS(-DYMAKE=1)
    PEERDIR(
        library/svnversion
    )
ELSE()
    SET_SVNREVISION()
ENDIF()

PEERDIR(
    library/getopt/small
)

SRCS(
    GLOBAL print.cpp
)

END()
