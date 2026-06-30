#pragma once

#include "socket.h"

// assume s is non-blocking, return non-blocking socket
SOCKET Accept4(SOCKET s, struct sockaddr* addr, socklen_t* addrlen);
// create non-blocking socket
SOCKET Socket4(int domain, int type, int protocol);
