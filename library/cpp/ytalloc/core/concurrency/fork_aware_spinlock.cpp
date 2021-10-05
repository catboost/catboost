#include "fork_aware_spinlock.h"
#include "rw_spinlock.h"

#ifdef _unix_
    #include <pthread.h>
#endif

#include <atomic>
#include <array>

namespace NYT::NConcurrency {

////////////////////////////////////////////////////////////////////////////////

static constexpr int MaxAtForkHandlers = 8;
using TAtForkHandler = TForkAwareSpinLock::TAtForkHandler;

struct TAtForkHandlerSet
{
    void* Cookie;
    TAtForkHandler Prepare;
    TAtForkHandler Parent;
    TAtForkHandler Child;
    std::atomic<bool> Initialized;
};

static std::array<TAtForkHandlerSet, MaxAtForkHandlers> AtForkHandlerSets;
static std::atomic<int> AtForkHandlerCount;

////////////////////////////////////////////////////////////////////////////////

class TForkProtector
{
public:
    static TForkProtector* Get()
    {
        static TForkProtector Instance;
        return &Instance;
    }

    TReaderWriterSpinLock& ForkLock()
    {
        return ForkLock_;
    }

private:
    TReaderWriterSpinLock ForkLock_;

    TForkProtector()
    {
#ifdef _unix_
        pthread_atfork(
            &TForkProtector::OnPrepare,
            &TForkProtector::OnParent,
            &TForkProtector::OnChild);
#endif
    }

    static void OnPrepare()
    {
        for (const auto& set : AtForkHandlerSets) {
            if (set.Initialized.load() && set.Prepare) {
                set.Prepare(set.Cookie);
            }
        }
        Get()->ForkLock().AcquireWriter();
    }

    static void OnParent()
    {
        Get()->ForkLock().ReleaseWriter();
        for (const auto& set : AtForkHandlerSets) {
            if (set.Initialized.load() && set.Parent) {
                set.Parent(set.Cookie);
            }
        }
    }

    static void OnChild()
    {
        Get()->ForkLock().ReleaseWriter();
        for (const auto& set : AtForkHandlerSets) {
            if (set.Initialized.load() && set.Child) {
                set.Child(set.Cookie);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

static struct TForkProtectorInitializer
{
    TForkProtectorInitializer()
    {
        TForkProtector::Get();
    }
} ForkProtectorInitializer;

////////////////////////////////////////////////////////////////////////////////

void TForkAwareSpinLock::Acquire() noexcept
{
    TForkProtector::Get()->ForkLock().AcquireReaderForkFriendly();
    SpinLock_.Acquire();
}

void TForkAwareSpinLock::Release() noexcept
{
    SpinLock_.Release();
    TForkProtector::Get()->ForkLock().ReleaseReader();
}

bool TForkAwareSpinLock::IsLocked() noexcept
{
    return SpinLock_.IsLocked();
}

void TForkAwareSpinLock::AtFork(
    void* cookie,
    TAtForkHandler prepare,
    TAtForkHandler parent,
    TAtForkHandler child)
{
    TForkProtector::Get();
    int index = AtForkHandlerCount++;
    Y_VERIFY(index < MaxAtForkHandlers);
    auto& set = AtForkHandlerSets[index];
    set.Cookie = cookie;
    set.Prepare = prepare;
    set.Parent = parent;
    set.Child = child;
    set.Initialized.store(true);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NConcurrency

