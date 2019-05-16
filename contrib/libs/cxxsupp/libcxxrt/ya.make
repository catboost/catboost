LIBRARY()

LICENSE(
    BSD
)



PEERDIR(contrib/libs/cppdemangle)

IF (CXX_UNWIND STREQUAL "glibcxx_dynamic" OR ARCH_PPC64LE)
    LDFLAGS(-lgcc_s)
ELSE()
    PEERDIR(contrib/libs/libunwind_master)
ENDIF()

ADDINCL(
    GLOBAL contrib/libs/cxxsupp/libcxxrt
)

IF (CLANG OR USE_LTO)
    PEERDIR(
        contrib/libs/cxxsupp/builtins
    )
ENDIF ()

NO_RUNTIME()
NO_COMPILER_WARNINGS()

IF (SANITIZER_TYPE STREQUAL undefined)
    NO_SANITIZE()
ENDIF ()

IF (MUSL)
    ADDINCL(
        contrib/libs/musl-1.1.20/arch/generic
        contrib/libs/musl-1.1.20/arch/x86_64
        contrib/libs/musl-1.1.20/extra
        contrib/libs/musl-1.1.20/include
    )
ENDIF ()

CXXFLAGS(
    -nostdinc++
)

SRCS(
    memory.cc
    auxhelper.cc
    stdexcept.cc
    exception.cc
    guard.cc
    typeinfo.cc
    dynamic_cast.cc
    unwind.cc
)

END()
