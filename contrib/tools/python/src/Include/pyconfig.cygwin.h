#pragma once

#include "pyconfig.linux.h"

#undef HAVE_LINUX_NETLINK_H
#undef HAVE_LINUX_TIPC_H
#undef HAVE_NETPACKET_PACKET_H
#undef HAVE_SYS_EPOLL_H
#undef HAVE_EPOLL
#undef HAVE_LIBINTL_H
#undef HAVE_GETRESGID
#undef HAVE_SETRESGID
#undef HAVE_GETRESUID
#undef HAVE_SETRESUID
#undef HAVE_GETLOADAVG
#undef HAVE_TMPNAM_R
#undef HAVE_MREMAP

#if defined(_64_)
#undef SIZEOF_FPOS_T
#define SIZEOF_FPOS_T 8

#undef SIZEOF_WCHAR_T
#define SIZEOF_WCHAR_T 2
#endif

#undef VA_LIST_IS_ARRAY
