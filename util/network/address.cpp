#include <util/stream/str.h>

#include "address.h"

#if defined(_unix_)
    #include <sys/un.h>
#endif

#ifndef UNIX_PATH_MAX
inline constexpr size_t UNIX_PATH_LIMIT = 108;
#else
inline constexpr size_t UNIX_PATH_LIMIT = UNIX_PATH_MAX;
#endif

using namespace NAddr;

template <bool printPort>
static inline void PrintAddr(IOutputStream& out, const IRemoteAddr& addr) {
    const sockaddr* a = addr.Addr();
    char buf[INET6_ADDRSTRLEN + 10];

    switch (a->sa_family) {
        case AF_INET: {
            const TIpAddress sa(*(const sockaddr_in*)a);

            out << IpToString(sa.Host(), buf, sizeof(buf));

            if (printPort) {
                out << ":" << sa.Port();
            }

            break;
        }

        case AF_INET6: {
            const sockaddr_in6* sa = (const sockaddr_in6*)a;

            if (!inet_ntop(AF_INET6, (void*)&sa->sin6_addr.s6_addr, buf, sizeof(buf))) {
                ythrow TSystemError() << "inet_ntop() failed";
            }

            if (printPort) {
                out << "[" << buf << "]"
                    << ":" << InetToHost(sa->sin6_port);
            } else {
                out << buf;
            }

            break;
        }

#if defined(AF_UNIX)
        case AF_UNIX: {
            const sockaddr_un* sa = (const sockaddr_un*)a;

            out << TStringBuf(sa->sun_path);

            break;
        }
#endif

        default: {
            size_t len = addr.Len();

            const char* b = (const char*)a;
            const char* e = b + len;

            bool allZeros = true;
            for (size_t i = 0; i < len; ++i) {
                if (b[i] != 0) {
                    allZeros = false;
                    break;
                }
            }

            if (allZeros) {
                out << "(raw all zeros)";
            } else {
                out << "(raw " << (int)a->sa_family << " ";

                while (b != e) {
                    //just print raw bytes
                    out << (int)*b++;
                    if (b != e) {
                        out << " ";
                    }
                }

                out << ")";
            }

            break;
        }
    }
}

template <>
void Out<IRemoteAddr>(IOutputStream& out, const IRemoteAddr& addr) {
    PrintAddr<true>(out, addr);
}

template <>
void Out<NAddr::TAddrInfo>(IOutputStream& out, const NAddr::TAddrInfo& addr) {
    PrintAddr<true>(out, addr);
}

template <>
void Out<NAddr::TIPv4Addr>(IOutputStream& out, const NAddr::TIPv4Addr& addr) {
    PrintAddr<true>(out, addr);
}

template <>
void Out<NAddr::TIPv6Addr>(IOutputStream& out, const NAddr::TIPv6Addr& addr) {
    PrintAddr<true>(out, addr);
}

template <>
void Out<NAddr::TOpaqueAddr>(IOutputStream& out, const NAddr::TOpaqueAddr& addr) {
    PrintAddr<true>(out, addr);
}

void NAddr::PrintHost(IOutputStream& out, const IRemoteAddr& addr) {
    PrintAddr<false>(out, addr);
}

TString NAddr::PrintHost(const IRemoteAddr& addr) {
    TStringStream ss;
    PrintAddr<false>(ss, addr);
    return ss.Str();
}

TString NAddr::PrintHostAndPort(const IRemoteAddr& addr) {
    TStringStream ss;
    PrintAddr<true>(ss, addr);
    return ss.Str();
}

IRemoteAddrPtr NAddr::GetSockAddr(SOCKET s) {
    auto addr = MakeHolder<TOpaqueAddr>();

    if (getsockname(s, addr->MutableAddr(), addr->LenPtr()) < 0) {
        ythrow TSystemError() << "getsockname() failed";
    }

    return addr;
}

IRemoteAddrPtr NAddr::GetPeerAddr(SOCKET s) {
    auto addr = MakeHolder<TOpaqueAddr>();

    if (getpeername(s, addr->MutableAddr(), addr->LenPtr()) < 0) {
        ythrow TSystemError() << "getpeername() failed";
    }

    return addr;
}

static const in_addr& InAddr(const IRemoteAddr& addr) {
    return ((const sockaddr_in*)addr.Addr())->sin_addr;
}

static const in6_addr& In6Addr(const IRemoteAddr& addr) {
    return ((const sockaddr_in6*)addr.Addr())->sin6_addr;
}

bool NAddr::IsLoopback(const IRemoteAddr& addr) {
    if (addr.Addr()->sa_family == AF_INET) {
        return ((ntohl(InAddr(addr).s_addr) >> 24) & 0xff) == 127;
    }

    if (addr.Addr()->sa_family == AF_INET6) {
        return 0 == memcmp(&In6Addr(addr), &in6addr_loopback, sizeof(in6_addr));
    }

    return false;
}

bool NAddr::IsSame(const IRemoteAddr& lhs, const IRemoteAddr& rhs) {
    if (lhs.Addr()->sa_family != rhs.Addr()->sa_family) {
        return false;
    }

    if (lhs.Addr()->sa_family == AF_INET) {
        return InAddr(lhs).s_addr == InAddr(rhs).s_addr;
    }

    if (lhs.Addr()->sa_family == AF_INET6) {
        return 0 == memcmp(&In6Addr(lhs), &In6Addr(rhs), sizeof(in6_addr));
    }

    ythrow yexception() << "unsupported addr family: " << lhs.Addr()->sa_family;
}

socklen_t NAddr::SockAddrLength(const sockaddr* addr) {
    switch (addr->sa_family) {
        case AF_INET:
            return sizeof(sockaddr_in);

        case AF_INET6:
            return sizeof(sockaddr_in6);

#if defined(AF_LOCAL)
        case AF_LOCAL:
            return sizeof(sockaddr_un);
#endif
    }

    ythrow yexception() << "unsupported address family: " << addr->sa_family;
}

TUnixSocketAddr::TUnixSocketAddr(TStringBuf path) {
    Y_ENSURE(path.Size() <= UNIX_PATH_LIMIT - 1,
             "Unix path \"" << path << "\" is longer than " << UNIX_PATH_LIMIT);
    SockAddr_.Set(path);
}

const sockaddr* TUnixSocketAddr::Addr() const {
    return SockAddr_.SockAddr();
}

socklen_t TUnixSocketAddr::Len() const {
    return SockAddr_.Len();
}
