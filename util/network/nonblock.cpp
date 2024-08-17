#include "nonblock.h"

#include <util/system/platform.h>

#include <util/generic/singleton.h>

#if defined(_unix_)
    #include <dlfcn.h>
#endif

#if defined(_linux_)
    #if !defined(SOCK_NONBLOCK)
        #define SOCK_NONBLOCK 04000
    #endif
#endif

namespace {
    struct TFeatureCheck {
        inline TFeatureCheck()
            : Accept4(nullptr)
            , HaveSockNonBlock(false)
        {
#if defined(_unix_) && defined(SOCK_NONBLOCK)
            {
                Accept4 = reinterpret_cast<TAccept4>(dlsym(RTLD_DEFAULT, "accept4"));

    #if defined(_musl_)
                // musl always statically linked
                if (!Accept4) {
                    Accept4 = accept4;
                }
    #endif

                if (Accept4) {
                    Accept4(-1, nullptr, nullptr, SOCK_NONBLOCK);

                    if (errno == ENOSYS) {
                        Accept4 = nullptr;
                    }
                }
            }
#endif

#if defined(SOCK_NONBLOCK)
            {
                TSocketHolder tmp(socket(PF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0));

                HaveSockNonBlock = !tmp.Closed();
            }
#endif
        }

        inline SOCKET FastAccept(SOCKET s, struct sockaddr* addr, socklen_t* addrlen) const {
#if defined(SOCK_NONBLOCK)
            if (Accept4) {
                return Accept4(s, addr, addrlen, SOCK_NONBLOCK);
            }
#endif

            const SOCKET ret = accept(s, addr, addrlen);

#if !defined(_freebsd_)
            // freebsd inherit O_NONBLOCK flag
            if (ret != INVALID_SOCKET) {
                SetNonBlock(ret);
            }
#endif

            return ret;
        }

        inline SOCKET FastSocket(int domain, int type, int protocol) const {
#if defined(SOCK_NONBLOCK)
            if (HaveSockNonBlock) {
                return socket(domain, type | SOCK_NONBLOCK, protocol);
            }
#endif

            const SOCKET ret = socket(domain, type, protocol);

            if (ret != INVALID_SOCKET) {
                SetNonBlock(ret);
            }

            return ret;
        }

        static inline const TFeatureCheck* Instance() noexcept {
            return Singleton<TFeatureCheck>();
        }

        using TAccept4 = int (*)(int sockfd, struct sockaddr* addr, socklen_t* addrlen, int flags);
        TAccept4 Accept4;
        bool HaveSockNonBlock;
    };
}

SOCKET Accept4(SOCKET s, struct sockaddr* addr, socklen_t* addrlen) {
    return TFeatureCheck::Instance()->FastAccept(s, addr, addrlen);
}

SOCKET Socket4(int domain, int type, int protocol) {
    return TFeatureCheck::Instance()->FastSocket(domain, type, protocol);
}
