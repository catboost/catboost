#include "platform-linux.hpp"

#undef ZMQ_HAVE_EVENTFD
#undef ZMQ_HAVE_SO_PEERCRED
#undef ZMQ_HAVE_SOCK_CLOEXEC
#undef ZMQ_HAVE_TCP_KEEPIDLE
#undef ZMQ_USE_EPOLL

#define ZMQ_USE_KQUEUE
