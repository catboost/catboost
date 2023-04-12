#pragma once

#include "ip.h"
#include "socket.h"

#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/network/sock.h>

namespace NAddr {
    class IRemoteAddr {
    public:
        virtual ~IRemoteAddr() = default;

        virtual const sockaddr* Addr() const = 0;
        virtual socklen_t Len() const = 0;
    };

    using IRemoteAddrPtr = THolder<IRemoteAddr>;
    using IRemoteAddrRef = TAtomicSharedPtr<NAddr::IRemoteAddr>;

    IRemoteAddrPtr GetSockAddr(SOCKET s);
    IRemoteAddrPtr GetPeerAddr(SOCKET s);
    void PrintHost(IOutputStream& out, const IRemoteAddr& addr);

    TString PrintHost(const IRemoteAddr& addr);
    TString PrintHostAndPort(const IRemoteAddr& addr);

    bool IsLoopback(const IRemoteAddr& addr);
    bool IsSame(const IRemoteAddr& lhs, const IRemoteAddr& rhs);

    socklen_t SockAddrLength(const sockaddr* addr);

    //for accept, recvfrom - see LenPtr()
    class TOpaqueAddr: public IRemoteAddr {
    public:
        inline TOpaqueAddr() noexcept
            : L_(sizeof(S_))
        {
            Zero(S_);
        }

        inline TOpaqueAddr(const IRemoteAddr* addr) noexcept {
            Assign(addr->Addr(), addr->Len());
        }

        inline TOpaqueAddr(const sockaddr* addr) {
            Assign(addr, SockAddrLength(addr));
        }

        const sockaddr* Addr() const override {
            return MutableAddr();
        }

        socklen_t Len() const override {
            return L_;
        }

        inline sockaddr* MutableAddr() const noexcept {
            return (sockaddr*)&S_;
        }

        inline socklen_t* LenPtr() noexcept {
            return &L_;
        }

    private:
        inline void Assign(const sockaddr* addr, socklen_t len) noexcept {
            L_ = len;
            memcpy(MutableAddr(), addr, L_);
        }

    private:
        sockaddr_storage S_;
        socklen_t L_;
    };

    //for TNetworkAddress
    class TAddrInfo: public IRemoteAddr {
    public:
        inline TAddrInfo(const addrinfo* ai) noexcept
            : AI_(ai)
        {
        }

        const sockaddr* Addr() const override {
            return AI_->ai_addr;
        }

        socklen_t Len() const override {
            return (socklen_t)AI_->ai_addrlen;
        }

    private:
        const addrinfo* const AI_;
    };

    //compat, for TIpAddress
    class TIPv4Addr: public IRemoteAddr {
    public:
        inline TIPv4Addr(const TIpAddress& addr) noexcept
            : A_(addr)
        {
        }

        const sockaddr* Addr() const override {
            return A_;
        }

        socklen_t Len() const override {
            return A_;
        }

    private:
        const TIpAddress A_;
    };

    //same, for ipv6 addresses
    class TIPv6Addr: public IRemoteAddr {
    public:
        inline TIPv6Addr(const sockaddr_in6& a) noexcept
            : A_(a)
        {
        }

        const sockaddr* Addr() const override {
            return (sockaddr*)&A_;
        }

        socklen_t Len() const override {
            return sizeof(A_);
        }

    private:
        const sockaddr_in6 A_;
    };

    class TUnixSocketAddr: public IRemoteAddr {
    public:
        explicit TUnixSocketAddr(TStringBuf path);

        const sockaddr* Addr() const override;

        socklen_t Len() const override;

    private:
        TSockAddrLocal SockAddr_;
    };
}
