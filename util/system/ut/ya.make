UNITTEST_FOR(util)



FORK_TESTS()

FORK_SUBTESTS()

SPLIT_FACTOR(40)

TIMEOUT(300)

SIZE(MEDIUM)

IF (OS_DARWIN)
    SIZE(LARGE)
    TAG(ya:fat ya:force_sandbox ya:exotic_platform)
    TIMEOUT(3600)
ENDIF()

PEERDIR(
    library/testing/benchmark
)

SRCS(
    system/align_ut.cpp
    system/atexit_ut.cpp
    system/atomic_ut.cpp
    system/backtrace_ut.cpp
    system/byteorder_ut.cpp
    system/compat_ut.cpp
    system/compiler_ut.cpp
    system/context_ut.cpp
    system/condvar_ut.cpp
    system/cpu_id_ut.cpp
    system/datetime_ut.cpp
    system/demangle_ut.cpp
    system/daemon_ut.cpp
    system/direct_io_ut.cpp
    system/env_ut.cpp
    system/error_ut.cpp
    system/event_ut.cpp
    system/execpath_ut.cpp
    system/file_ut.cpp
    system/filemap_ut.cpp
    system/flock_ut.cpp
    system/fs_ut.cpp
    system/fstat_ut.cpp
    system/getpid_ut.cpp
    system/guard_ut.cpp
    system/hi_lo_ut.cpp
    system/hostname_ut.cpp
    system/info_ut.cpp
    system/interrupt_signals_ut.cpp
    system/mem_info_ut.cpp
    system/mincore_ut.cpp
    system/mutex_ut.cpp
    system/nice_ut.cpp
    system/pipe_ut.cpp
    system/platform_ut.cpp
    system/progname_ut.cpp
    system/rusage_ut.cpp
    system/rwlock_ut.cpp
    system/sanitizers_ut.cpp
    system/shellcommand_ut.cpp
    system/spinlock_ut.cpp
    system/src_root_ut.cpp
    system/src_location_ut.cpp
    system/shmat_ut.cpp
    system/tempfile_ut.cpp
    system/thread_ut.cpp
    system/tls_ut.cpp
    system/types_ut.cpp
    system/user_ut.cpp
    system/unaligned_mem_ut.cpp
    system/yassert_ut.cpp
)

IF (OS_WINDOWS)
    DEPENDS(
        util/system/ut/stdin_osfhandle
    )
ENDIF()

END()
