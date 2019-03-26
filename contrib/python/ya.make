

RECURSE(
    appdirs
    atomicwrites
    attrs
    dateutil
    dateutil/tests
    enum34
    funcsigs
    Jinja2
    MarkupSafe
    more-itertools
    numpy
    packaging
    pandas
    pandas/matplotlib
    pathlib2
    pluggy
    py
    pyparsing
    pytest
    pytz
    pytz/tests
    scandir
    setuptools
    six
    subprocess32
)

IF (OS_DARWIN)
    RECURSE(
    
)
ENDIF ()

IF (OS_LINUX)
    RECURSE(
    
)

    IF (NOT OS_SDK STREQUAL "ubuntu-12")
        RECURSE(
    
)
    ENDIF()
ENDIF ()
