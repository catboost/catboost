DLL(access_spreader_native_symbolic)

INCLUDE(${ARCADIA_ROOT}/library/cpp/yt/ya_cpp.make.inc)

SUBSCRIBER(g:yt)

SRCS(
    access_spreader_native.cpp
)

PEERDIR(
    library/cpp/yt/system
)

LDFLAGS(-Wl,-Bsymbolic)

END()
