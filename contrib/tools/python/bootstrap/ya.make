

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/python/bootstrap/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    PROGRAM()

    INCLUDE(${ARCADIA_ROOT}/contrib/tools/python/pyconfig.inc)

    PEERDIR(
        ${PYTHON_DIR}/base
    )

    ADDINCL(
        ${PYTHON_SRC_DIR}
        ${PYTHON_SRC_DIR}/Include
    )

    SRCDIR(
        ${PYTHON_SRC_DIR}
    )

    CFLAGS(${PYTHON_FLAGS} -DLIBDIR="${PYTHON_SRC_ROOT}/Lib" -DPYLIB="${PYTHON_SRC_DIR}/Lib")

    SRCS(
        Python/frozen.c
        python.c
        vars.cpp
    )

    END()
ENDIF()
