LIBRARY()



IF (OS_WINDOWS)
    #PASS
ELSE ()
    NO_COMPILER_WARNINGS()

    COPY_FILE(subprocess32.py subprocess.py)

    PY_SRCS(
        TOP_LEVEL
        subprocess32.py
        subprocess.py
    )

    SRCS(
        _posixsubprocess.c
    )

    PY_REGISTER(_posixsubprocess)
ENDIF ()

NO_LINT()

END()
