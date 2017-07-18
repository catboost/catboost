LIBRARY(pexpect)



PEERDIR(
    contrib/python/ptyprocess-0.5
)

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

NO_LINT()

END()
