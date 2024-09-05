#pragma once

#include <util/system/defaults.h>

namespace NResolver {
    // resolve hostname and fills up to *slots slots in ip array;
    // actual number of slots filled is returned in *slots;
    int GetHostIP(const char* hostname, ui32* ip, size_t* slots);
    int GetDnsError();

    inline int GetHostIP(const char* hostname, ui32* ip) {
        size_t slots = 1;

        return GetHostIP(hostname, ip, &slots);
    }
} // namespace NResolver
