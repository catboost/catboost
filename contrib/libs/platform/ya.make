LIBRARY()


NO_LIBC()

PEERDIR(
    contrib/libs/platform/mode
)

CFLAGS(
    GLOBAL -D__STDC_CONSTANT_MACROS
    GLOBAL -D__STDC_FORMAT_MACROS
)

IF (MSVC)
    CFLAGS(
        GLOBAL -D_USE_MATH_DEFINES
    )
ENDIF ()

IF (UNIX)
    CFLAGS(
        GLOBAL -D_THREAD_SAFE
        GLOBAL -D_PTHREADS
        GLOBAL -D_REENTRANT
        GLOBAL -D_GNU_SOURCE
        GLOBAL -D_FILE_OFFSET_BITS=64
        GLOBAL -D_LARGEFILE_SOURCE
    )

    EXTRALIBS(-lc -lm -ldl)

    IF (GCC)
        # EXTRACOMPILERLIBS(libgcc.a)
    ENDIF ()

    IF (CLANG OR GCC)
        EXTRALIBS(-rdynamic)
    ENDIF ()

    # GOOGLE_SANITIZER_TYPE={memory,address,thread}
    IF (CLANG)
        IF (GOOGLE_SANITIZERS)
            EXTRALIBS(-fPIC -pie)
            EXTRALIBS(-fsanitize=${GOOGLE_SANITIZER_TYPE})
        ENDIF ()
    ENDIF ()

    IF (GCC)
        EXTRALIBS(-nodefaultlibs)
    ENDIF ()
ENDIF ()

IF (WINE)
    CFLAGS(
        GLOBAL -D_WIN64
        GLOBAL -DO_SEQUENTIAL=0010000
    )
ENDIF ()

IF (GCC)
    CFLAGS(
        GLOBAL -W
        GLOBAL -Wall
        GLOBAL -Wno-parentheses
    )
ENDIF ()

IF (CLANG)
    CFLAGS(
        GLOBAL -W
        GLOBAL -Wall
        GLOBAL -Wno-parentheses
        GLOBAL -Wno-invalid-source-encoding
    )

    # Experimental sanitizer support for arc-based clang builds
    IF (GOOGLE_SANITIZERS)
        CFLAGS(
            GLOBAL -fPIC # for msan
            GLOBAL -fPIE # msan + gdb = love
            GLOBAL -g
            GLOBAL -O1 # too slow on -O2 sometimes
            GLOBAL -fno-omit-frame-pointer
            GLOBAL -fsanitize=${GOOGLE_SANITIZER_TYPE}
            GLOBAL -fsanitize-blacklist=${GOOGLE_SANITIZER_BLACKLIST}
            GLOBAL -I${CLANG_INCLUDE_PATH}
        )
        # slower execution but shows uninitialized memory origins
        IF (GOOGLE_SANITIZER_TRACK_ORIGINS)
            CFLAGS(GLOBAL -fsanitize-memory-track-origins)
        ENDIF ()
    ELSE ()
        CFLAGS(
            GLOBAL -nostdinc++
        )
    ENDIF ()
ENDIF ()

END()
