LIBRARY()

LICENSE(MIT BSD-2-Clause)



PEERDIR(contrib/libs/cppdemangle)

IF (CXX_UNWIND == "glibcxx_dynamic" OR ARCH_PPC64LE)
    LDFLAGS(-lgcc_s)
ELSE()
    PEERDIR(contrib/libs/libunwind)
ENDIF()

ADDINCL(
    GLOBAL contrib/libs/cxxsupp/libcxxrt
)

NO_RUNTIME()
NO_COMPILER_WARNINGS()

IF (SANITIZER_TYPE == undefined)
    NO_SANITIZE()
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
