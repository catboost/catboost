

RECURSE(
    appdirs
    backports-shutil_get_terminal_size
    colorama
    dateutil
    decorator
    enum34
    ipython
    ipython/bin
    ipython_genutils
    Jinja2
    MarkupSafe
    numpy
    packaging
    pandas
    pandas/matplotlib
    path.py
    pathlib2
    pexpect
    pickleshare
    prompt_toolkit
    ptyprocess
    py
    Pygments
    pyparsing
    pytest
    pytz
    pytz/tests
    PyYAML
    requests
    setuptools
    simplegeneric-0.8.1
    six
    traitlets
    wcwidth-0.1.6
)

IF (OS_DARWIN)
    RECURSE(
    appnope
)
ENDIF ()

IF (OS_LINUX)
    RECURSE(
    
)
ENDIF ()
