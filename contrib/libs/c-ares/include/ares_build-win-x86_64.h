#pragma once

#define CARES_HAVE_WINSOCK2_H 
#ifdef CARES_HAVE_WINSOCK2_H
#  include <winsock2.h>
#endif

#define CARES_HAVE_WS2TCPIP_H 
#ifdef CARES_HAVE_WS2TCPIP_H
#  include <ws2tcpip.h>
#endif

#define CARES_HAVE_WINDOWS_H 
#ifdef CARES_HAVE_WINDOWS_H
#  include <windows.h>
#endif

#define CARES_TYPEOF_ARES_SOCKLEN_T int
typedef CARES_TYPEOF_ARES_SOCKLEN_T ares_socklen_t;

#define CARES_TYPEOF_ARES_SSIZE_T __int64
typedef CARES_TYPEOF_ARES_SSIZE_T ares_ssize_t;
