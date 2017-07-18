LIBRARY()

LICENSE(
    MIT
    BSD
)



ADDINCL(
    GLOBAL contrib/libs/cxxsupp/libcxx/include/wrappers
)

IF (OS_ANDROID)
    CXXFLAGS(
        -D__GLIBCXX__=1
    )

    SRCS(
        src/support/android/locale_android.cpp
    )

    PEERDIR(
        contrib/libs/cxxsupp/android
    )
    ADDINCL(
        GLOBAL contrib/libs/cxxsupp/android/include
    )
ELSEIF (OS_IOS)
    LDFLAGS(-lc++abi)
ELSEIF (CLANG OR MUSL OR OS_DARWIN OR USE_LTO)
    PEERDIR(
        ADDINCL contrib/libs/cxxsupp/libcxxrt
    )

    CXXFLAGS(
        -DLIBCXXRT=1
    )
    IF (MUSL)
        ADDINCL(
            GLOBAL contrib/libs/musl-1.1.16/arch/x86_64
            GLOBAL contrib/libs/musl-1.1.16/arch/generic
            GLOBAL contrib/libs/musl-1.1.16/include
            GLOBAL contrib/libs/musl-1.1.16/extra
        )
    ENDIF()
ELSEIF (OS_WINDOWS)
    PEERDIR(
        contrib/libs/pthreads_win32
    )

    SRCS(
        src/support/win32/support.cpp
        src/support/win32/locale_win32.cpp
        src/support/win32/time_win32.cpp
    )

    CFLAGS(
        GLOBAL -DY_STD_NAMESPACE=ystd
        GLOBAL -DNStl=Y_STD_NAMESPACE
        GLOBAL -Dstd=Y_STD_NAMESPACE
        GLOBAL -D_LIBCPP_VASPRINTF_DEFINED
        GLOBAL -D_WCHAR_H_CPLUSPLUS_98_CONFORMANCE_
    )
ELSE()
    LDFLAGS(-Wl,-Bstatic -lsupc++ -lgcc -lgcc_eh -Wl,-Bdynamic)

    CXXFLAGS(
        -D__GLIBCXX__=1
        -Wno-unknown-pragmas
    )
ENDIF ()

IF (OS_LINUX)
    EXTRALIBS(-lpthread)
ENDIF ()

IF (OS_FREEBSD)
    EXTRALIBS(-lthr)
    PEERDIR(
        contrib/libs/cxxsupp/old-freebsd
    )
ENDIF ()

IF (OS_DARWIN AND CLANG)
    # Standalone Clang for Darwin already contains libc++ and it breaks being used twice
    CFLAGS(GLOBAL -nostdinc++)
ENDIF()

NO_UTIL()
NO_RUNTIME()
NO_COMPILER_WARNINGS()

SRCS(
    src/debug.cpp
    src/regex.cpp
    src/optional.cpp
    src/shared_mutex.cpp
    src/mutex.cpp
    src/ios.cpp
    src/stdexcept.cpp
    src/new.cpp
    src/condition_variable.cpp
    src/typeinfo.cpp
    src/memory.cpp
    src/string.cpp
    src/system_error.cpp
    src/utility.cpp
    src/bind.cpp
    src/strstream.cpp
    src/valarray.cpp
    src/iostream.cpp
    src/future.cpp
    src/chrono.cpp
    src/thread.cpp
    src/hash.cpp
    src/algorithm.cpp
    src/locale.cpp
    src/exception.cpp
    src/random.cpp
)

END()
