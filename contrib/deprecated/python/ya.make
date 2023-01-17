

RECURSE(
    backports.functools-lru-cache
    backports.shutil-get-terminal-size
    faulthandler
    subprocess32
)

IF (OS_WINDOWS)
    RECURSE(
    win-unicode-console
)
ENDIF()
