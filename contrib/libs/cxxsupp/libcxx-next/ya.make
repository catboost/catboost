LIBRARY()

LICENSE(
    MIT
    BSD
)



ADDINCL(
    GLOBAL contrib/libs/cxxsupp/libcxx-next/include
)

CXXFLAGS(-D_LIBCPP_BUILDING_LIBRARY)

IF (OS_ANDROID)
    DEFAULT(CXX_RT "glibcxx_driver")
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
    DEFAULT(CXX_RT "libcxxrt")
    IF (MUSL)
        ADDINCL(
            GLOBAL contrib/libs/musl-1.1.16/arch/x86_64
            GLOBAL contrib/libs/musl-1.1.16/arch/generic
            GLOBAL contrib/libs/musl-1.1.16/include
            GLOBAL contrib/libs/musl-1.1.16/extra
        )
    ENDIF()
ELSEIF (OS_WINDOWS)
    SRCS(
        src/support/win32/locale_win32.cpp
        src/support/win32/support.cpp
        src/support/win32/atomic_win32.cpp
        src/support/win32/new_win32.cpp
        src/support/win32/threading_support_win32.cpp
    )

    CFLAGS(
        GLOBAL -DY_STD_NAMESPACE=ystd
        GLOBAL -DNStl=Y_STD_NAMESPACE
        GLOBAL -Dstd=Y_STD_NAMESPACE
        GLOBAL -D_LIBCPP_VASPRINTF_DEFINED
        GLOBAL -D_WCHAR_H_CPLUSPLUS_98_CONFORMANCE_
    )
ELSE()
    DEFAULT(CXX_RT "glibcxx_static")
    CXXFLAGS(
        -Wno-unknown-pragmas
        -nostdinc++
    )
ENDIF ()

IF (OS_LINUX)
    EXTRALIBS(-lpthread)
ENDIF ()

IF (CLANG)
    CFLAGS(GLOBAL -nostdinc++)
ENDIF()

# The CXX_RT variable controls which C++ runtime is used.
# * libcxxrt        - https://github.com/pathscale/libcxxrt library stored in Arcadia
# * glibcxx         - GNU C++ Library runtime with default (static) linkage
# * glibcxx_static  - GNU C++ Library runtime with static linkage
# * glibcxx_dynamic - GNU C++ Library runtime with dynamic linkage
# * glibcxx_driver  - GNU C++ Library runtime provided by the compiler driver
# * default         - default C++ runtime provided by the compiler driver
#
# All glibcxx* runtimes are taken from system/compiler SDK

DEFAULT(CXX_RT "default")

IF (CXX_RT STREQUAL "libcxxrt")
    PEERDIR(ADDINCL contrib/libs/cxxsupp/libcxxrt)
    CXXFLAGS(-DLIBCXXRT=1)
ELSEIF (CXX_RT STREQUAL "glibcxx" OR CXX_RT STREQUAL "glibcxx_static")
    LDFLAGS(-Wl,-Bstatic -lsupc++ -lgcc -lgcc_eh -Wl,-Bdynamic)
    CXXFLAGS(-D__GLIBCXX__=1)
ELSEIF (CXX_RT STREQUAL "glibcxx_dynamic")
    LDFLAGS(-lgcc_s -lstdc++)
    CXXFLAGS(-D__GLIBCXX__=1)
ELSEIF (CXX_RT STREQUAL "glibcxx_driver")
    CXXFLAGS(-D__GLIBCXX__=1)
ELSEIF (CXX_RT STREQUAL "default")
    # Do nothing
ELSE()
    MESSAGE(FATAL_ERROR "Unexpected CXX_RT value: ${CXX_RT}")
ENDIF()

NO_UTIL()
NO_RUNTIME()
NO_COMPILER_WARNINGS()

SRCS(
    src/algorithm.cpp
    src/any.cpp
    src/bind.cpp
    src/chrono.cpp
    src/condition_variable.cpp
    src/debug.cpp
    src/exception.cpp
    src/functional.cpp
    src/future.cpp
    src/hash.cpp
    src/ios.cpp
    src/iostream.cpp
    src/locale.cpp
    src/memory.cpp
    src/mutex.cpp
    src/optional.cpp
    src/random.cpp
    src/regex.cpp
    src/shared_mutex.cpp
    src/stdexcept.cpp
    src/string.cpp
    src/strstream.cpp
    src/system_error.cpp
    src/thread.cpp
    src/utility.cpp
    src/valarray.cpp
    src/variant.cpp
    src/vector.cpp
)

IF(NOT OS_WINDOWS)
    SRCS(
        src/new.cpp
        src/typeinfo.cpp
    )
ENDIF()

END()
