

RECURSE(
    backports.functools-lru-cache
    backports.shutil-get-terminal-size
    configparser
    enum34
    faulthandler
    scandir
    subprocess32
    typing
)

IF (OS_WINDOWS)
    RECURSE(
    win-unicode-console
)
ENDIF()
