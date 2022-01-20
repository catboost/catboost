#include "utils.h"

#include <util/generic/utility.h>
#include <util/stream/output.h>
#include <util/stream/str.h>
#include <util/system/error.h>

#if defined(_unix_)
#include <pthread.h>
#endif

#if defined(_win_)
#include <windows.h>
#endif

using namespace NNeh;

size_t NNeh::RealStackSize(size_t len) noexcept {
#if defined(NDEBUG) && !defined(_san_enabled_)
    return len;
#else
    return Max<size_t>(len, 64000);
#endif
}

TString NNeh::PrintHostByRfc(const NAddr::IRemoteAddr& addr) {
    TStringStream ss;

    if (addr.Addr()->sa_family == AF_INET) {
        NAddr::PrintHost(ss, addr);
    } else if (addr.Addr()->sa_family == AF_INET6) {
        ss << '[';
        NAddr::PrintHost(ss, addr);
        ss << ']';
    }
    return ss.Str();
}

NAddr::IRemoteAddrPtr NNeh::GetPeerAddr(SOCKET s) {
    TAutoPtr<NAddr::TOpaqueAddr> addr(new NAddr::TOpaqueAddr());

    if (getpeername(s, addr->MutableAddr(), addr->LenPtr()) < 0) {
        ythrow TSystemError() << "getpeername() failed";
    }

    return NAddr::IRemoteAddrPtr(addr.Release());
}
