

RECURSE(
    appdirs
    atomicwrites
    attrs
    dateutil
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
ENDIF ()
