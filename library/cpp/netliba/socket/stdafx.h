#pragma once

#include <util/system/platform.h>
#if defined(_darwin_)
#define __APPLE_USE_RFC_2292
#endif

#include <util/system/compat.h>
#include <util/network/init.h>
#if defined(_unix_)
#include <netdb.h>
#include <fcntl.h>
#elif defined(_win_)
#include <winsock2.h>
using socklen_t = int;
#endif
