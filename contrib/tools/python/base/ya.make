LIBRARY()

LICENSE(
    PSF
)



NO_WSHADOW()
ENABLE(NO_WIN32_LEAN)

CFLAGS(
    GLOBAL -DARCADIA_PYTHON_UNICODE_SIZE=${ARCADIA_PYTHON_UNICODE_SIZE}
)

IF (NOT MSVC)
    CFLAGS(
        -fwrapv
    )
ENDIF()

SRCDIR(
    contrib/tools/python/src/Include
)

INCLUDE(${ARCADIA_ROOT}/contrib/tools/python/pyconfig.inc)
INCLUDE(CMakeLists.inc)

IF (YMAKE)
    CHECK_CONFIG_H(pyconfig.h)
ENDIF ()

END()
