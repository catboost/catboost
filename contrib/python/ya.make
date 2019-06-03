

RECURSE(
    atomicwrites
    attrs
    dateutil
    enum34
    funcsigs
    graphviz
    Jinja2
    Jinja2/examples
    MarkupSafe
    more-itertools
    numpy
    pandas
    pandas/matplotlib
    pathlib2
    pluggy
    py
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
