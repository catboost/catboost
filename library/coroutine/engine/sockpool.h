#pragma once

#include "impl.h"
#include "network.h"

#include <util/network/address.h>
#include <util/network/socket.h>
#include <util/system/mutex.h>

extern void SetCommonSockOpts(SOCKET sock, const struct sockaddr* sa = nullptr);

class TSocketPool;

class TPooledSocket {
    class TImpl: public TIntrusiveListItem<TImpl>, public TSimpleRefCount<TImpl, TImpl> {
    public:
        TImpl(SOCKET fd, TSocketPool* pool) noexcept
            : Pool_(pool)
            , IsKeepAlive_(false)
            , Fd_(fd)
        {
            Touch();
        }

        static void Destroy(TImpl* impl) noexcept {
            impl->DoDestroy();
        }

        void DoDestroy() noexcept {
            if (!Closed() && IsKeepAlive() && IsInGoodState()) {
                ReturnToPool();
            } else {
                delete this;
            }
        }

        bool IsKeepAlive() const noexcept {
            return IsKeepAlive_;
        }

        void SetKeepAlive(bool ka) {
            ::SetKeepAlive(Fd_, ka);
            IsKeepAlive_ = ka;
        }

        SOCKET Socket() const noexcept {
            return Fd_;
        }

        bool Closed() const noexcept {
            return Fd_.Closed();
        }

        void Close() noexcept {
            Fd_.Close();
        }

        bool IsInGoodState() const noexcept {
            int err = 0;
            socklen_t len = sizeof(err);

            getsockopt(Fd_, SOL_SOCKET, SO_ERROR, (char*)&err, &len);

            return !err;
        }

        bool IsOpen() const noexcept {
            return IsInGoodState() && IsNotSocketClosedByOtherSide(Fd_);
        }

        void Touch() noexcept {
            TouchTime_ = TInstant::Now();
        }

        const TInstant& LastTouch() const noexcept {
            return TouchTime_;
        }

    private:
        inline void ReturnToPool() noexcept;

    private:
        TSocketPool* Pool_;
        bool IsKeepAlive_;
        TSocketHolder Fd_;
        TInstant TouchTime_;
    };

    friend class TSocketPool;

public:
    TPooledSocket()
        : Impl_(nullptr)
    {
    }

    TPooledSocket(TImpl* impl)
        : Impl_(impl)
    {
    }

    ~TPooledSocket() {
        if (UncaughtException() && !!Impl_) {
            Close();
        }
    }

    operator SOCKET() const noexcept {
        return Impl_->Socket();
    }

    void SetKeepAlive(bool ka) {
        Impl_->SetKeepAlive(ka);
    }

    void Close() noexcept {
        Impl_->Close();
    }

private:
    TIntrusivePtr<TImpl> Impl_;
};

struct TConnectData {
    TConnectData(TCont* cont, const TInstant& deadLine)
        : Cont(cont)
        , DeadLine(deadLine)
    {
    }

    TConnectData(TCont* cont, const TDuration& timeOut)
        : Cont(cont)
        , DeadLine(TInstant::Now() + timeOut)
    {
    }

    TCont* Cont;
    const TInstant DeadLine;
};

class TSocketPool {
    friend class TPooledSocket::TImpl;

public:
    typedef TAtomicSharedPtr<NAddr::IRemoteAddr> TAddrRef;

    TSocketPool(int ip, int port)
        : Addr_(new NAddr::TIPv4Addr(TIpAddress((ui32)ip, (ui16)port)))
    {
    }

    TSocketPool(const TAddrRef& addr)
        : Addr_(addr)
    {
    }

    void EraseStale(const TInstant& maxAge) noexcept {
        TSockets toDelete;

        {
            TGuard<TMutex> guard(Mutex_);

            for (TSockets::TIterator it = Pool_.Begin(); it != Pool_.End();) {
                if (it->LastTouch() < maxAge) {
                    toDelete.PushBack(&*(it++));
                } else {
                    ++it;
                }
            }
        }
    }

    TPooledSocket Get(TConnectData* conn) {
        TPooledSocket ret;

        if (TPooledSocket::TImpl* alive = GetImpl()) {
            ret = TPooledSocket(alive);
        } else {
            ret = AllocateMore(conn);
        }

        ret.Impl_->Touch();

        return ret;
    }

    bool GetAlive(TPooledSocket& socket) {
        if (TPooledSocket::TImpl* alive = GetImpl()) {
            alive->Touch();
            socket = TPooledSocket(alive);
            return true;
        }
        return false;
    }

private:
    TPooledSocket::TImpl* GetImpl() {
        TGuard<TMutex> guard(Mutex_);

        while (!Pool_.Empty()) {
            THolder<TPooledSocket::TImpl> ret(Pool_.PopFront());

            if (ret->IsOpen()) {
                return ret.Release();
            }
        }
        return nullptr;
    }

    void Release(TPooledSocket::TImpl* impl) noexcept {
        TGuard<TMutex> guard(Mutex_);

        Pool_.PushFront(impl);
    }

    TPooledSocket AllocateMore(TConnectData* conn);

private:
    TAddrRef Addr_;
    using TSockets = TIntrusiveListWithAutoDelete<TPooledSocket::TImpl, TDelete>;
    TSockets Pool_;
    TMutex Mutex_;
};

inline void TPooledSocket::TImpl::ReturnToPool() noexcept {
    Pool_->Release(this);
}


class TContIO: public IInputStream, public IOutputStream {
public:
    TContIO(SOCKET fd, TCont* cont)
        : Fd_(fd)
        , Cont_(cont)
    {
    }

    void DoWrite(const void* buf, size_t len) override {
        NCoro::WriteI(Cont_, Fd_, buf, len).Checked();
    }

    size_t DoRead(void* buf, size_t len) override {
        return NCoro::ReadI(Cont_, Fd_, buf, len).Checked();
    }

    SOCKET Fd() const noexcept {
        return Fd_;
    }

private:
    SOCKET Fd_;
    TCont* Cont_;
};
