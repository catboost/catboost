PY_LIBRARY() # Backport from Python 3.



LICENSE(PSFv2)

VERSION(3.5.4)

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

    PY_REGISTER(_posixsubprocess32)

    PY_SRCS(
        TOP_LEVEL
        subprocess.py
    )
ENDIF ()

NO_LINT()

END()

RECURSE_FOR_TESTS(
    testdata
)
