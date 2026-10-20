GTEST()

INCLUDE(${ARCADIA_ROOT}/library/cpp/yt/ya_cpp.make.inc)

SUBSCRIBER(g:yt)

SRCS(
    access_spreader_dynamic_tls_ut.cpp
)

PEERDIR(
    library/cpp/testing/common
    library/cpp/testing/gtest
)

DEPENDS(
    library/cpp/yt/system/unittests/access_spreader_dynamic_tls/native_symbolic
)

ALLOCATOR(SYSTEM)

END()

RECURSE(
    native_symbolic
)
