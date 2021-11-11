

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
    pandas
    parso
    path.py
    pathlib2
    pexpect
    pickleshare
    pluggy
    prompt-toolkit
    ptyprocess
    py
    Pygments
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
    tornado
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
