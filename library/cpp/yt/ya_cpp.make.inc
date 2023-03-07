# This file should be included in all YT projects (including YT ORM installations).

IF (NOT MSVC)
    CXXFLAGS(
        -Wdeprecated-this-capture
    )
ENDIF()
