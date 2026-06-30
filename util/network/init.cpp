#include "init.h"

#include <util/system/defaults.h>
#include <util/generic/singleton.h>

namespace {
    class TNetworkInit {
    public:
        inline TNetworkInit() {
#ifndef ROBOT_SIGPIPE
            signal(SIGPIPE, SIG_IGN);
#endif

#if defined(_win_)
    #pragma comment(lib, "ws2_32.lib")
            WSADATA wsaData;
            int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
            Y_ASSERT(!result);
            if (result) {
                exit(-1);
            }
#endif
        }
    };
} // namespace

void InitNetworkSubSystem() {
    (void)Singleton<TNetworkInit>();
}
