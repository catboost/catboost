LIBRARY()



LICENSE(PSFv2)

VERSION(3.2.7)

COPY_FILE(subprocess32.py subprocess.py)

PY_SRCS(
    TOP_LEVEL
    subprocess32.py
    subprocess.py
)

IF (OS_WINDOWS)
    #PASS
ELSE ()
    NO_COMPILER_WARNINGS()

    SRCS(
        _posixsubprocess.c
    )

    PY_REGISTER(_posixsubprocess)
ENDIF ()

NO_LINT()

END()
