#include "waitable_registry.h"
#include "local_executor.h"

#include <util/generic/intrlist.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>
#include <util/system/atomic.h>
#include <util/system/spinlock.h>
#include <util/system/spin_wait.h>

#include <utility>

namespace {
    struct TRegistrable : public TIntrusiveListItem<TRegistrable>,
                          public ::NPar::ILocallyExecutable {
    };

    class TRegistry : public ::NPar::IWaitableRegistry {
    public:
        TIntrusivePtr<::NPar::ILocallyExecutable>
        Register(::NPar::TLocallyExecutableFunction) override;

        TIntrusivePtr<::NPar::ILocallyExecutable>
        Register(TIntrusivePtr<::NPar::ILocallyExecutable>) override;

        void Wait() override;
        void Reset() override;

        void DoRegister(TRegistrable*);
        void DoUnregister(TRegistrable*);

    private:
        TAdaptiveLock ThisLock_;

        TAdaptiveLock RegistryLock_;
        TIntrusiveList<TRegistrable> Registry_;
    };

    // Wrapers have a member with intrusive pointer to the registry that created them. We are doing
    // this to make sure that registry is still existing even if thread that owned the registry
    // destroyed it.

    class TFunctionWrapper : public TRegistrable {
    public:
        TFunctionWrapper(
            TIntrusivePtr<TRegistry> parent,
            ::NPar::TLocallyExecutableFunction f);
        void LocalExec(int id) override;

    private:
        TIntrusivePtr<TRegistry> Registry_;
        ::NPar::TLocallyExecutableFunction Slave_;
    };

    class TFunctorWrapper : public TRegistrable {
    public:
        TFunctorWrapper(
            TIntrusivePtr<TRegistry> parent,
            TIntrusivePtr<::NPar::ILocallyExecutable> executable);
        void LocalExec(int id) override;

    private:
        TIntrusivePtr<TRegistry> Registry_;
        TIntrusivePtr<::NPar::ILocallyExecutable> Slave_;
    };
}

TIntrusivePtr<::NPar::IWaitableRegistry> NPar::MakeDefaultWaitableRegistry() {
    return MakeIntrusive<TRegistry>();
}

TFunctionWrapper::TFunctionWrapper(
    TIntrusivePtr<TRegistry> r,
    ::NPar::TLocallyExecutableFunction f)
    : Registry_{std::move(r)}
    , Slave_{std::move(f)} {
}

TFunctorWrapper::TFunctorWrapper(
    TIntrusivePtr<TRegistry> r,
    TIntrusivePtr<::NPar::ILocallyExecutable> e)
    : Registry_{std::move(r)}
    , Slave_{std::move(e)} {
}

void TRegistry::DoRegister(TRegistrable* item) {
    with_lock (RegistryLock_) {
        Registry_.PushBack(item);
    }
}

void TRegistry::DoUnregister(TRegistrable* item) {
    with_lock (RegistryLock_) {
        item->Unlink();
    }
}

TIntrusivePtr<::NPar::ILocallyExecutable>
TRegistry::Register(::NPar::TLocallyExecutableFunction f) {
    with_lock (ThisLock_) {
        auto res = MakeIntrusive<TFunctionWrapper>(this, std::move(f));
        DoRegister(res.Get());
        return res;
    }
}

TIntrusivePtr<::NPar::ILocallyExecutable>
TRegistry::Register(TIntrusivePtr<::NPar::ILocallyExecutable> e) {
    with_lock (ThisLock_) {
        if (auto* const r = dynamic_cast<TRegistrable*>(e.Get())) {
            // There may be a race condition if this item is already registred somewhere and is
            // already executing. But this race is OK, because code is already malformed and we are
            // only increasing chances to catch it by throwing exception on invalid condition.
            //
            const auto alreadyRegistred = !r->Empty();
            Y_ENSURE(!alreadyRegistred);

            // We already have a wrapper, lets just register it.
            DoRegister(r);
            return e;
        }

        auto res = MakeIntrusive<TFunctorWrapper>(this, std::move(e));
        DoRegister(res.Get());
        return res;
    }
}

void TRegistry::Wait() {
    with_lock (ThisLock_) {
        for (TSpinWait sw;;) {
            with_lock (RegistryLock_) {
                if (Registry_.Empty()) {
                    return;
                }
            }
            sw.Sleep();
        }
    }
}

void TRegistry::Reset() {
    with_lock (ThisLock_) {
        with_lock (RegistryLock_) {
            Registry_.Clear();
        }
    }
}

namespace {
    class TGuard {
    public:
        TGuard(TRegistry* registry, TRegistrable* item)
            : Registry_{registry}
            , Item_{item} {
        }

        ~TGuard() {
            Registry_->DoUnregister(Item_);
        }

    private:
        TRegistry* const Registry_;
        TRegistrable* const Item_;
    };
}

void TFunctionWrapper::LocalExec(const int id) {
    TGuard g{Registry_.Get(), this};
    Slave_(id);
}

void TFunctorWrapper::LocalExec(const int id) {
    TGuard g{Registry_.Get(), this};
    Slave_->LocalExec(id);
}
