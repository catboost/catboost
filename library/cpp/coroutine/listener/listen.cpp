#include "listen.h"

#include <library/cpp/coroutine/engine/impl.h>
#include <library/cpp/coroutine/engine/network.h>

#include <util/network/ip.h>
#include <util/network/address.h>
#include <util/generic/ylimits.h>
#include <util/generic/intrlist.h>

using namespace NAddr;

namespace {
    union TSa {
        const sockaddr* Sa;
        const sockaddr_in* In;
        const sockaddr_in6* In6;

        inline TSa(const sockaddr* sa) noexcept
            : Sa(sa)
        {
        }

        inline bool operator==(const TSa& r) const noexcept {
            if (Sa->sa_family == r.Sa->sa_family) {
                switch (Sa->sa_family) {
                    case AF_INET:
                        return In->sin_port == r.In->sin_port && In->sin_addr.s_addr == r.In->sin_addr.s_addr;
                    case AF_INET6:
                        return In6->sin6_port == r.In6->sin6_port && !memcmp(&In6->sin6_addr, &r.In6->sin6_addr, sizeof(in6_addr));
                }
            }

            return false;
        }

        inline bool operator!=(const TSa& r) const noexcept {
            return !(*this == r);
        }
    };
}

class TContListener::TImpl {
private:
    struct TStoredAddrInfo: public TAddrInfo, private TNetworkAddress {
        inline TStoredAddrInfo(const struct addrinfo* ai, const TNetworkAddress& addr) noexcept
            : TAddrInfo(ai)
            , TNetworkAddress(addr)
        {
        }
    };

private:
    class TOneSocketListener: public TIntrusiveListItem<TOneSocketListener> {
    public:
        inline TOneSocketListener(TImpl* parent, IRemoteAddrPtr addr)
            : Parent_(parent)
            , C_(nullptr)
            , ListenSocket_(socket(addr->Addr()->sa_family, SOCK_STREAM, 0))
            , Addr_(std::move(addr))
        {
            if (ListenSocket_ == INVALID_SOCKET) {
                ythrow TSystemError() << "can not create socket";
            }

            FixIPv6ListenSocket(ListenSocket_);
            CheckedSetSockOpt(ListenSocket_, SOL_SOCKET, SO_REUSEADDR, 1, "reuse addr");

            const TOptions& opts = Parent_->Opts_;
            if (opts.SendBufSize) {
                SetOutputBuffer(ListenSocket_, opts.SendBufSize);
            }
            if (opts.RecvBufSize) {
                SetInputBuffer(ListenSocket_, opts.RecvBufSize);
            }
            if (opts.ReusePort) {
                SetReusePort(ListenSocket_, opts.ReusePort);
            }

            SetNonBlock(ListenSocket_);

            if (bind(ListenSocket_, Addr_->Addr(), Addr_->Len()) < 0) {
                ythrow TSystemError() << "bind failed";
            }
        }

        inline ~TOneSocketListener() {
            Stop();
        }

    public:
        inline void Run(TCont* cont) noexcept {
            C_ = cont;
            DoRun();
            C_ = nullptr;
        }

        inline void StartListen() {
            if (!C_) {
                const TOptions& opts = Parent_->Opts_;

                if (listen(ListenSocket_, (int)Min<size_t>(Max<int>(), opts.ListenQueue)) < 0) {
                    ythrow TSystemError() << "listen failed";
                }

                if (opts.EnableDeferAccept) {
                    SetDeferAccept(ListenSocket_);
                }

                C_ = Parent_->E_->Create<TOneSocketListener, &TOneSocketListener::Run>(this, "listen_job");
            }
        }

        inline const IRemoteAddr* Addr() const noexcept {
            return Addr_.Get();
        }

        inline void Stop() noexcept {
            if (C_) {
                C_->Cancel();

                while (C_) {
                    Parent_->E_->Running()->Yield();
                }
            }
        }

    private:
        inline void DoRun() noexcept {
            while (!C_->Cancelled()) {
                try {
                    TOpaqueAddr remote;
                    const int res = NCoro::AcceptI(C_, ListenSocket_, remote.MutableAddr(), remote.LenPtr());

                    if (res < 0) {
                        const int err = -res;

                        if (err != ECONNABORTED) {
                            if (err == ECANCELED) {
                                break;
                            }
                            if (errno == EMFILE) {
                                C_->SleepT(TDuration::MilliSeconds(1));
                            }

                            ythrow TSystemError(err) << "can not accept";
                        }
                    } else {
                        TSocketHolder c((SOCKET)res);

                        const ICallBack::TAcceptFull acc = {
                            &c,
                            &remote,
                            Addr(),
                        };

                        Parent_->Cb_->OnAcceptFull(acc);
                    }
                } catch (...) {
                    try {
                        Parent_->Cb_->OnError();
                    } catch (...) {
                    }
                }
            }

            try {
                Parent_->Cb_->OnStop(&ListenSocket_);
            } catch (...) {
            }
        }

    private:
        const TImpl* const Parent_;
        TCont* C_;
        TSocketHolder ListenSocket_;
        const IRemoteAddrPtr Addr_;
    };

private:
    class TListeners: public TIntrusiveListWithAutoDelete<TOneSocketListener, TDelete> {
    private:
        template <class T>
        using TIt = std::conditional_t<std::is_const<T>::value, typename T::TConstIterator, typename T::TIterator>;

        template <class T>
        static inline TIt<T> FindImpl(T* t, const IRemoteAddr& addr) {
            const TSa sa(addr.Addr());

            TIt<T> it = t->Begin();
            TIt<T> const end = t->End();

            while (it != end && sa != it->Addr()->Addr()) {
                ++it;
            }

            return it;
        }

    public:
        inline TIterator Find(const IRemoteAddr& addr) {
            return FindImpl(this, addr);
        }

        inline TConstIterator Find(const IRemoteAddr& addr) const {
            return FindImpl(this, addr);
        }
    };

public:
    inline TImpl(ICallBack* cb, TContExecutor* e, const TOptions& opts) noexcept
        : E_(e)
        , Cb_(cb)
        , Opts_(opts)
    {
    }

    inline void Listen() {
        for (TListeners::TIterator it = L_.Begin(); it != L_.End(); ++it) {
            it->StartListen();
        }
    }

    inline void Listen(const IRemoteAddr& addr) {
        const TListeners::TIterator it = L_.Find(addr);

        if (it != L_.End()) {
            it->StartListen();
        }
    }

    inline void Bind(const IRemoteAddr& addr) {
        const TSa sa(addr.Addr());

        switch (sa.Sa->sa_family) {
            case AF_INET:
                L_.PushBack(new TOneSocketListener(this, MakeHolder<TIPv4Addr>(*sa.In)));
                break;
            case AF_INET6:
                L_.PushBack(new TOneSocketListener(this, MakeHolder<TIPv6Addr>(*sa.In6)));
                break;
            default:
                ythrow yexception() << TStringBuf("unknown protocol");
        }
    }

    inline void Bind(const TIpAddress& addr) {
        L_.PushBack(new TOneSocketListener(this, MakeHolder<TIPv4Addr>(addr)));
    }

    inline void Bind(const TNetworkAddress& addr) {
        for (TNetworkAddress::TIterator it = addr.Begin(); it != addr.End(); ++it) {
            L_.PushBack(new TOneSocketListener(this, MakeHolder<TStoredAddrInfo>(&*it, addr)));
        }
    }

    inline void StopListenAddr(const IRemoteAddr& addr) {
        const TListeners::TIterator it = L_.Find(addr);

        if (it != L_.End()) {
            delete &*it;
        }
    }

private:
    TContExecutor* const E_;
    ICallBack* const Cb_;
    TListeners L_;
    const TOptions Opts_;
};

TContListener::TContListener(ICallBack* cb, TContExecutor* e, const TOptions& opts)
    : Impl_(new TImpl(cb, e, opts))
{
}

TContListener::~TContListener() {
}

namespace {
    template <class T>
    static inline T&& CheckImpl(T&& impl) {
        Y_ENSURE_EX(impl, yexception() << "not running");
        return std::forward<T>(impl);
    }
}

void TContListener::Listen(const IRemoteAddr& addr) {
    CheckImpl(Impl_)->Listen(addr);
}

void TContListener::Listen(const TIpAddress& addr) {
    return Listen(TIPv4Addr(addr));
}

void TContListener::Listen(const TNetworkAddress& addr) {
    for (TNetworkAddress::TIterator it = addr.Begin(); it != addr.End(); ++it) {
        Listen(TAddrInfo(&*it));
    }
}

void TContListener::Listen() {
    CheckImpl(Impl_)->Listen();
}

void TContListener::Bind(const IRemoteAddr& addr) {
    CheckImpl(Impl_)->Bind(addr);
}

void TContListener::Bind(const TIpAddress& addr) {
    return Bind(TIPv4Addr(addr));
}

void TContListener::Bind(const TNetworkAddress& addr) {
    CheckImpl(Impl_)->Bind(addr);
}

void TContListener::Stop() noexcept {
    Impl_.Destroy();
}

void TContListener::StopListenAddr(const IRemoteAddr& addr) {
    CheckImpl(Impl_)->StopListenAddr(addr);
}

void TContListener::StopListenAddr(const TIpAddress& addr) {
    return StopListenAddr(TIPv4Addr(addr));
}

void TContListener::StopListenAddr(const TNetworkAddress& addr) {
    for (TNetworkAddress::TIterator it = addr.Begin(); it != addr.End(); ++it) {
        StopListenAddr(TAddrInfo(&*it));
    }
}

void TContListener::ICallBack::OnAcceptFull(const TAcceptFull& params) {
    const TSa remote(params.Remote->Addr());
    const TSa local(params.Local->Addr());

    if (local.Sa->sa_family == AF_INET) {
        const TIpAddress r(*remote.In);
        const TIpAddress l(*local.In);

        const TAccept a = {
            params.S, &r, &l};

        OnAccept(a);
    }
}

void TContListener::ICallBack::OnStop(TSocketHolder* s) {
    s->ShutDown(SHUT_RDWR);
    s->Close();
}
