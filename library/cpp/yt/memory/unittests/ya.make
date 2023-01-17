GTEST(unittester-library-memory)



IF (NOT OS_WINDOWS)
    ALLOCATOR(YT)
ENDIF()

SRCS(
    atomic_intrusive_ptr_ut.cpp
    intrusive_ptr_ut.cpp
    weak_ptr_ut.cpp
)

PEERDIR(
    library/cpp/testing/gtest
    library/cpp/yt/memory
)

END()
