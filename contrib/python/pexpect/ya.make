PY23_LIBRARY()

LICENSE(ISC)



VERSION(4.8.0)

PEERDIR(
    contrib/python/ptyprocess
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    pexpect/ANSI.py
    pexpect/FSM.py
    pexpect/__init__.py
    pexpect/exceptions.py
    pexpect/expect.py
    pexpect/fdpexpect.py
    pexpect/popen_spawn.py
    pexpect/pty_spawn.py
    pexpect/pxssh.py
    pexpect/replwrap.py
    pexpect/run.py
    pexpect/screen.py
    pexpect/spawnbase.py
    pexpect/utils.py
)

IF (PYTHON3)
    PY_SRCS(
        TOP_LEVEL
        pexpect/_async.py
    )
ENDIF()

RESOURCE_FILES(
    PREFIX contrib/python/pexpect/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
