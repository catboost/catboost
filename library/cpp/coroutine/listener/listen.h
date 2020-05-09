#pragma once

#include <util/generic/ptr.h>
#include <util/generic/ylimits.h>

struct TIpAddress;
class TContExecutor;
class TSocketHolder;
class TNetworkAddress;

namespace NAddr {
    class IRemoteAddr;
}

class TContListener {
public:
    struct TOptions {
        inline TOptions() noexcept
            : ListenQueue(Max<size_t>())
            , SendBufSize(0)
            , RecvBufSize(0)
            , EnableDeferAccept(false)
            , ReusePort(false)
        {
        }

        inline TOptions& SetListenQueue(size_t len) noexcept {
            ListenQueue = len;

            return *this;
        }

        inline TOptions& SetDeferAccept(bool enable) noexcept {
            EnableDeferAccept = enable;

            return *this;
        }

        inline TOptions& SetSendBufSize(unsigned size) noexcept {
            SendBufSize = size;

            return *this;
        }

        inline TOptions& SetRecvBufSize(unsigned size) noexcept {
            RecvBufSize = size;

            return *this;
        }

        inline TOptions& SetReusePort(bool reusePort) noexcept {
            ReusePort = reusePort;

            return *this;
        }

        size_t ListenQueue;
        unsigned SendBufSize;
        unsigned RecvBufSize;
        bool EnableDeferAccept;
        bool ReusePort;
    };

    class ICallBack {
    public:
        struct TAccept {
            TSocketHolder* S;
            const TIpAddress* Remote;
            const TIpAddress* Local;
        };

        struct TAcceptFull {
            TSocketHolder* S;
            const NAddr::IRemoteAddr* Remote;
            const NAddr::IRemoteAddr* Local;
        };

        virtual void OnAccept(const TAccept&) {
        }

        virtual void OnAcceptFull(const TAcceptFull&);

        /*
         * will be called from catch (...) {} context
         * so your can re-throw current exception and work around it
         */
        virtual void OnError() = 0;

        virtual void OnStop(TSocketHolder*);

        virtual ~ICallBack() {
        }
    };

    TContListener(ICallBack* cb, TContExecutor* e, const TOptions& opts = TOptions());
    ~TContListener();

    /// start listener threads
    void Listen();

    void Listen(const NAddr::IRemoteAddr& addr);
    void Listen(const TIpAddress& addr);
    void Listen(const TNetworkAddress& addr);

    /// bind server on address. Can be called multiple times to bind on more then one address
    void Bind(const NAddr::IRemoteAddr& addr);
    void Bind(const TIpAddress& addr);
    void Bind(const TNetworkAddress& addr);

    void Stop() noexcept;

    void StopListenAddr(const NAddr::IRemoteAddr& addr);
    void StopListenAddr(const TIpAddress& addr);
    void StopListenAddr(const TNetworkAddress& addr);

    template <class T>
    inline void StartListenAddr(const T& addr) {
        Bind(addr);
        Listen(addr);
    }

private:
    class TImpl;
    THolder<TImpl> Impl_;
};
