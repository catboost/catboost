#include "endpoint.h"
#include "sock.h"

TEndpoint::TEndpoint(const TEndpoint::TAddrRef& addr)
    : Addr_(addr)
{
    const sockaddr* sa = Addr_->Addr();

    if (sa->sa_family != AF_INET && sa->sa_family != AF_INET6 && sa->sa_family != AF_UNIX) {
        ythrow yexception() << TStringBuf("endpoint can contain only ipv4, ipv6 or unix address");
    }
}

TEndpoint::TEndpoint()
    : Addr_(new NAddr::TIPv4Addr(TIpAddress(TIpHost(0), TIpPort(0))))
{
}

void TEndpoint::SetPort(ui16 port) {
    if (Port() == port || Addr_->Addr()->sa_family == AF_UNIX) {
        return;
    }

    NAddr::TOpaqueAddr* oa = new NAddr::TOpaqueAddr(Addr_.Get());
    Addr_.Reset(oa);
    sockaddr* sa = oa->MutableAddr();

    if (sa->sa_family == AF_INET) {
        ((sockaddr_in*)sa)->sin_port = HostToInet(port);
    } else {
        ((sockaddr_in6*)sa)->sin6_port = HostToInet(port);
    }
}

ui16 TEndpoint::Port() const noexcept {
    if (Addr_->Addr()->sa_family == AF_UNIX) {
        return 0;
    }

    const sockaddr* sa = Addr_->Addr();

    if (sa->sa_family == AF_INET) {
        return InetToHost(((const sockaddr_in*)sa)->sin_port);
    } else {
        return InetToHost(((const sockaddr_in6*)sa)->sin6_port);
    }
}

size_t TEndpoint::Hash() const {
    const sockaddr* sa = Addr_->Addr();

    if (sa->sa_family == AF_INET) {
        const sockaddr_in* sa4 = (const sockaddr_in*)sa;

        return IntHash((((ui64)sa4->sin_addr.s_addr) << 16) ^ sa4->sin_port);
    } else if (sa->sa_family == AF_INET6) {
        const sockaddr_in6* sa6 = (const sockaddr_in6*)sa;
        const ui64* ptr = (const ui64*)&sa6->sin6_addr;

        return IntHash(ptr[0] ^ ptr[1] ^ sa6->sin6_port);
    } else {
        const sockaddr_un* un = (const sockaddr_un*)sa;
        THash<TString> strHash;

        return strHash(un->sun_path);
    }
}
