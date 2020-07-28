#pragma once

#include "cstdafx.h"

#include <util/system/compat.h>
#include <util/network/init.h>
#if defined(_unix_)
#include <netdb.h>
#include <fcntl.h>
#elif defined(_win_)
#include <winsock2.h>
using socklen_t = int;
#endif

#include <util/generic/ptr.h>

template <class T>
static const T* BreakAliasing(const void* f) {
    return (const T*)f;
}

template <class T>
static T* BreakAliasing(void* f) {
    return (T*)f;
}
