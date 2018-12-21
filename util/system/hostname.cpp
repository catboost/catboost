#include <util/memory/tempbuf.h>
#include <util/generic/string.h>
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>
#include <util/network/ip.h>

#if defined(_unix_)
#include <unistd.h>
#include <ifaddrs.h>
#include <netdb.h>
#endif

#if defined(_win_)
#include <WinSock2.h>
#endif

#include "defaults.h"
#include "yassert.h"
#include "hostname.h"

namespace {
    struct THostNameHolder {
        inline THostNameHolder() {
            TTempBuf hostNameBuf;

            if (gethostname(hostNameBuf.Data(), hostNameBuf.Size() - 1)) {
                ythrow TSystemError() << "can not get host name";
            }

            HostName = hostNameBuf.Data();
        }

        TString HostName;
    };

    struct TFQDNHostNameHolder {
        inline TFQDNHostNameHolder() {
            struct addrinfo hints;
            struct addrinfo* ais{nullptr};
            char buf[1024];

            memset(buf, 0, sizeof(buf));
            int res = gethostname(buf, sizeof(buf) - 1);
            if (res) {
                ythrow TSystemError() << "can not get hostname";
            }

            memset(&hints, 0, sizeof(hints));
            hints.ai_family = AF_UNSPEC;
            hints.ai_flags = AI_CANONNAME;
            res = getaddrinfo(buf, nullptr, &hints, &ais);
            if (res) {
                ythrow TSystemError() << "can not get FQDN (return code is " << res << ", hostname is \"" << buf << "\")";
            }
            FQDNHostName = ais->ai_canonname;
            FQDNHostName.to_lower();
            freeaddrinfo(ais);
        }

        TString FQDNHostName;
    };
}

const TString& HostName() {
    return (Singleton<THostNameHolder>())->HostName;
}

const char* GetHostName() {
    return HostName().data();
}

const TString& FQDNHostName() {
    return (Singleton<TFQDNHostNameHolder>())->FQDNHostName;
}

const char* GetFQDNHostName() {
    return FQDNHostName().data();
}

bool IsFQDN(const TString& name) {
    TString absName = name;
    if (!absName.EndsWith('.')) {
        absName.append(".");
    }

    try {
        // ResolveHost() can't be used since it is ipv4-only, port is not important
        TNetworkAddress addr(absName, 0);
    } catch (const TNetworkResolutionError&) {
        return false;
    }
    return true;
}
