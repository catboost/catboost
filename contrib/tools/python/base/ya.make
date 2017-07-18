LIBRARY()



NO_WSHADOW()
ENABLE(NO_WIN32_LEAN)

SRCDIR(
    contrib/tools/python/src
)

IF (OS_DARWIN)
    ADDINCL(
        contrib/tools/python/base/darwin
    )
ENDIF ()

INCLUDE(${ARCADIA_ROOT}/contrib/tools/python/pyconfig.inc)
INCLUDE(CMakeLists.inc)

IF (YMAKE)
    CHECK_CONFIG_H(pyconfig.h)
ENDIF ()

END()
