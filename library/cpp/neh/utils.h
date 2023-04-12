#pragma once

#include <library/cpp/threading/atomic/bool.h>

#include <util/network/address.h>
#include <util/system/thread.h>
#include <util/generic/cast.h>
#include <library/cpp/deprecated/atomic/atomic.h>

namespace NNeh {
    typedef TAutoPtr<TThread> TThreadRef;

    template <typename T, void (T::*M)()>
    static void* HelperMemberFunc(void* arg) {
        T* obj = reinterpret_cast<T*>(arg);
        (obj->*M)();
        return nullptr;
    }

    template <typename T, void (T::*M)()>
    static TThreadRef Spawn(T* t) {
        TThreadRef thr(new TThread(HelperMemberFunc<T, M>, t));

        thr->Start();

        return thr;
    }

    size_t RealStackSize(size_t len) noexcept;

    //from rfc3986:
    //host          = IP-literal / IPv4address / reg-name
    //IP-literal    = "[" ( IPv6address / IPvFuture  ) "]"
    TString PrintHostByRfc(const NAddr::IRemoteAddr& addr);

    NAddr::IRemoteAddrPtr GetPeerAddr(SOCKET s);

    using TAtomicBool = NAtomic::TBool;
}
