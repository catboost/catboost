

RECURSE(
    atomicwrites
    attrs
    configparser
    contextlib2
    dateutil
    enum34
    faulthandler
    funcsigs
    graphviz
    importlib-metadata
    Jinja2
    MarkupSafe
    more-itertools
    numpy
    pandas
    pathlib2
    pluggy
    py
    pytest
    pytz
    scandir
    scipy
    setuptools
    six
    subprocess32
    testpath
)

IF (OS_WINDOWS)
    RECURSE(
    
)
ENDIF()

IF (OS_DARWIN)
    RECURSE(
    
)
ENDIF ()

IF (OS_LINUX)
    RECURSE(
    
)

    IF (OS_SDK != "ubuntu-12")
        RECURSE(
    
)
    ENDIF()
ENDIF ()
