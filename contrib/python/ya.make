

RECURSE(
    appdirs
    dateutil
    enum34
    Jinja2
    MarkupSafe
    numpy
    packaging
    pandas
    pandas/matplotlib
    py
    pyparsing
    pytest
    pytz
    pytz/tests
    PyYAML
    setuptools
    six
)

IF (OS_DARWIN)
    RECURSE(
    
)
ENDIF ()

IF (OS_LINUX)
    RECURSE(
    
)
ENDIF ()
