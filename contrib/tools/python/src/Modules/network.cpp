#include <util/network/socket.h>

extern "C" {
    int IsReusePortAvailableFromUtil() {
        return IsReusePortAvailable() ? 1 : 0;
    }
}

