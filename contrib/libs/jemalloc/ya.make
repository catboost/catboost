LIBRARY()



VERSION(3.6.0)

LICENSE(BSD2)

NO_COMPILER_WARNINGS()

NO_UTIL()

ADDINCL(
    contrib/libs/jemalloc/include
    contrib/libs/libunwind/include
)

IF (OS_WINDOWS)
    ADDINCL(
        contrib/libs/jemalloc/include/msvc_compat
    )
ELSEIF (OS_DARWIN OR OS_IOS)
    SRCS(
        src/zone.c
        GLOBAL reg_zone.cpp
    )
ELSE ()
    CFLAGS(
        -fvisibility=hidden
    )
ENDIF ()

CFLAGS(
    -funroll-loops
)

SRCS(
    src/arena.c
    src/atomic.c
    src/base.c
    src/bitmap.c
    src/chunk.c
    src/chunk_dss.c
    src/chunk_mmap.c
    src/ckh.c
    src/ctl.c
    src/extent.c
    src/hash.c
    src/huge.c
    src/jemalloc.c
    src/mb.c
    src/mutex.c
    src/prof.c
    src/quarantine.c
    src/rtree.c
    src/stats.c
    src/tcache.c
    src/tsd.c
    src/util.c
    hack.cpp
)

END()
