

RECURSE(
    atomicwrites
    attrs
    backcall
    backports.functools-lru-cache
    backports.shutil-get-terminal-size
    colorama
    configparser
    contextlib2
    dateutil
    decorator
    enum34
    faulthandler
    filelock
    funcsigs
    graphviz
    importlib-metadata
    iniconfig
    ipdb
    ipython
    ipython-genutils
    jedi
    Jinja2
    joblib
    MarkupSafe
    matplotlib-inline
    more-itertools
    mypy-protobuf
    numpy
    packaging
    pandas
    parso
    pathlib2
    pexpect
    pickleshare
    pluggy
    prompt-toolkit
    protobuf
    ptyprocess
    py
    Pygments
    pyparsing
    pytest
    pytz
    scandir
    scikit-learn
    scipy
    setuptools
    simplegeneric
    six
    subprocess32
    testpath
    tomli
    tornado
    traitlets
    typing
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
