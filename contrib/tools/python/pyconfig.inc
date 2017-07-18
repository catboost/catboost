NO_COMPILER_WARNINGS()
NO_UTIL()
NO_JOIN_SRC()

ENABLE(USE_ARCADIA_PYTHON)

SET(PYTHON_DIR "contrib/tools/python")
SET(PYTHON_SRC_DIR "${PYTHON_DIR}/src")

SET(PYTHON_ROOT "${ARCADIA_ROOT}/${PYTHON_DIR}")
SET(PYTHON_SRC_ROOT "${ARCADIA_ROOT}/${PYTHON_SRC_DIR}")

SET(PYTHON_FLAGS "-DPy_BUILD_CORE")

CFLAGS(
    -DUNQUOTED_VERSION=2.7
)

IF (NOT WIN32)
    EXTRALIBS(-lutil)

    IF (LINUX)
        EXTRALIBS(-ldl)

        CFLAGS(
            -DUNQUOTED_PYTHONPATH=:plat-linux2
            -DUNQUOTED_PLATFORM=linux2
        )
    ELSEIF (FREEBSD)
        CFLAGS(
            -DUNQUOTED_PYTHONPATH=:plat-freebsd${FREEBSD_VER}
            -DUNQUOTED_PLATFORM=freebsd${FREEBSD_VER}
        )
    ELSEIF (DARWIN)
        CFLAGS(
            -DUNQUOTED_PYTHONPATH=:plat-darwin
            -DUNQUOTED_PLATFORM=darwin
        )
        #SET_APPEND(OBJADDE
        #    -framework SystemConfiguration
        #    -framework CoreFoundation
        #)
    ENDIF ()
ENDIF ()

IF (MSVC)
    CFLAGS(/DXMLCALL= /DXMLIMPORT=)
ENDIF ()
