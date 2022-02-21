LIBRARY()

LICENSE(
    MIT AND
    NCSA
)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

VERSION(5.0)



NO_UTIL()

NO_PLATFORM()

NO_COMPILER_WARNINGS()

IF (SANITIZER_TYPE == thread)
    NO_SANITIZE()
    CFLAGS(
        -fPIC
    )
ENDIF()

IF (SANITIZER_TYPE == memory)
    NO_SANITIZE()
    CFLAGS(
        -fPIC
    )
ENDIF()

COMPILE_C_AS_CXX()

CXXFLAGS(-fno-exceptions)

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
    asm.S
    kmp_affinity.cpp
    kmp_alloc.c
    kmp_atomic.c
    kmp_barrier.cpp
    kmp_cancel.cpp
    kmp_csupport.c
    kmp_debug.c
    kmp_dispatch.cpp
    kmp_environment.c
    kmp_error.c
    kmp_ftn_cdecl.c
    kmp_ftn_extra.c
    kmp_global.c
    kmp_gsupport.c
    kmp_i18n.c
    kmp_io.c
    kmp_itt.c
    kmp_lock.cpp
    kmp_runtime.c
    kmp_sched.cpp
    kmp_settings.c
    kmp_str.c
    kmp_taskdeps.cpp
    kmp_tasking.c
    kmp_taskq.c
    kmp_threadprivate.c
    kmp_utility.c
    kmp_version.c
    kmp_wait_release.cpp
    #ompt-general.c
    thirdparty/ittnotify/ittnotify_static.c
    z_Linux_util.c
)

END()
