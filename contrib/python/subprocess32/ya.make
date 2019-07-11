LIBRARY()



LICENSE(PSFv2)

VERSION(3.2.7)

COPY_FILE(subprocess32.py subprocess.py)

PY_SRCS(
    TOP_LEVEL
    subprocess32.py
)

IF (NOT OS_WINDOWS)
    NO_COMPILER_WARNINGS()

    SRCS(
        _posixsubprocess.c
    )

    PY_REGISTER(_posixsubprocess)

    PY_SRCS(
        TOP_LEVEL
        subprocess.py
    )
ENDIF ()

NO_LINT()

END()
