LIBRARY()

LICENSE(
    MIT
    BSD
)



NO_UTIL()
NO_PLATFORM()
NO_COMPILER_WARNINGS()

IF (SANITIZER_TYPE STREQUAL thread)
    NO_SANITIZE()

    CFLAGS(-fPIC)
ENDIF ()

IF (SANITIZER_TYPE STREQUAL memory)
    NO_SANITIZE()

    CFLAGS(-fPIC)
ENDIF ()

COMPILE_C_AS_CXX()

CXXFLAGS(
    -fno-exceptions
)

SET_APPEND(CFLAGS -fno-lto)

ADDINCL(
    GLOBAL contrib/libs/cxxsupp/openmp
)

ADDINCL(
    contrib/libs/cxxsupp/openmp/i18n
    contrib/libs/cxxsupp/openmp/include/41
    contrib/libs/cxxsupp/openmp/thirdparty/ittnotify
)

SRCS(
    kmp_alloc.c
    kmp_atomic.c
    kmp_csupport.c
    kmp_debug.c
    kmp_itt.c
    kmp_environment.c
    kmp_error.c
    kmp_global.c
    kmp_i18n.c
    kmp_io.c
    kmp_runtime.c
    kmp_settings.c
    kmp_str.c
    kmp_tasking.c
    kmp_taskq.c
    kmp_threadprivate.c
    kmp_utility.c
)

SRCS(
    z_Linux_util.c
    kmp_gsupport.c
    asm.S
)

SRCS(
    thirdparty/ittnotify/ittnotify_static.c
)

SRCS(
    kmp_barrier.cpp
    kmp_wait_release.cpp
    kmp_affinity.cpp
    kmp_dispatch.cpp
    kmp_lock.cpp
    kmp_sched.cpp
    kmp_taskdeps.cpp
    kmp_cancel.cpp
)

SRCS(
    kmp_ftn_cdecl.c
    kmp_ftn_extra.c
    kmp_version.c
    #ompt-general.c
)

END()
