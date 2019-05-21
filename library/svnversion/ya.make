LIBRARY()

CREATE_SVNVERSION_FOR(
    svn_interface.c
    svnversion_data.h
)

SRCS(
    svnversion.cpp
    svn_interface.c
)
END()
