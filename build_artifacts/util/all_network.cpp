#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#endif
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif

#include "util/network/address.cpp"
#include "util/network/endpoint.cpp"
#include "util/network/hostip.cpp"
#include "util/network/init.cpp"
#include "util/network/interface.cpp"
#include "util/network/iovec.cpp"
#include "util/network/ip.cpp"
#include "util/network/nonblock.cpp"
#include "util/network/pair.cpp"
#include "util/network/poller.cpp"
#include "util/network/pollerimpl.cpp"
#include "util/network/sock.cpp"
#include "util/network/socket.cpp"
