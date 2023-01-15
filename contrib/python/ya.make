

RECURSE(
    atomicwrites
    attrs
    backports-shutil_get_terminal_size
    colorama
    configparser
    contextlib2
    dateutil
    decorator
    enum34
    faulthandler
    funcsigs
    graphviz
    importlib-metadata
    ipdb
    ipython
    ipython_genutils
    Jinja2
    MarkupSafe
    more-itertools
    numpy
    pandas
    path.py
    pathlib2
    pexpect
    pickleshare
    pluggy
    prompt_toolkit
    ptyprocess
    py
    Pygments
    pytest
    pytz
    scandir
    scipy
    setuptools
    simplegeneric
    six
    subprocess32
    testpath
    traitlets
    wcwidth
)

IF (OS_WINDOWS)
    RECURSE(
    win_unicode_console
)
ENDIF()

IF (OS_DARWIN)
    RECURSE(
    appnope
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
