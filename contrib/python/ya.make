

RECURSE(
    appdirs
    backports-shutil_get_terminal_size
    colorama-0.3.6
    dateutil
    decorator-4.0.6
    enum34
    ipython
    ipython/bin
    ipython_genutils-0.1.0
    Jinja2
    MarkupSafe
    numpy-1.11.1
    packaging
    pandas
    pandas/matplotlib
    path.py
    pathlib2
    pexpect-4.0.1
    pickleshare-0.6
    prompt_toolkit
    ptyprocess-0.5
    py-1.4.30
    Pygments
    pyparsing
    pytest
    pytz
    pytz/tests
    PyYAML-3.11
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
