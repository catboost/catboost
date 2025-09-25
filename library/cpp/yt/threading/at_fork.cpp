#include "at_fork.h"

#include <library/cpp/yt/memory/leaky_singleton.h>

#include <library/cpp/yt/assert/assert.h>

#ifdef _unix_
    #include <pthread.h>
#endif

#include <atomic>
#include <array>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

class TAtForkManager
{
public:
    static TAtForkManager* Get()
    {
        return LeakySingleton<TAtForkManager>();
    }

    void RegisterAtForkHandlers(
        TAtForkHandler prepare,
        TAtForkHandler parent,
        TAtForkHandler child)
    {
        int index = AtForkHandlerCount_++;
        Y_ABORT_UNLESS(index < MaxAtForkHandlerSets);
        auto& set = AtForkHandlerSets_[index];
        set.Prepare = std::move(prepare);
        set.Parent = std::move(parent);
        set.Child = std::move(child);
        set.Initialized.store(true);
    }

    TReaderWriterSpinLock* GetForkLock()
    {
        return &ForkLock_;
    }

private:
    DECLARE_LEAKY_SINGLETON_FRIEND()

    TReaderWriterSpinLock ForkLock_;

    struct TAtForkHandlerSet
    {
        TAtForkHandler Prepare;
        TAtForkHandler Parent;
        TAtForkHandler Child;
        std::atomic<bool> Initialized;
    };

    static constexpr int MaxAtForkHandlerSets = 8;
    std::array<TAtForkHandlerSet, MaxAtForkHandlerSets> AtForkHandlerSets_;
    std::atomic<int> AtForkHandlerCount_ = 0;

    TAtForkManager()
    {
#ifdef _unix_
        pthread_atfork(
            [] { Get()->OnPrepare(); },
            [] { Get()->OnParent(); },
            [] { Get()->OnChild(); });
#endif
    }

    void OnPrepare()
    {
        IterateAtForkHandlerSets([] (const TAtForkHandlerSet& set) {
            if (set.Prepare) {
                set.Prepare();
            }
        });
        ForkLock_.AcquireWriter();
    }


    void OnParent()
    {
        ForkLock_.ReleaseWriter();
        IterateAtForkHandlerSets([] (const TAtForkHandlerSet& set) {
            if (set.Parent) {
                set.Parent();
            }
        });
    }

    void OnChild()
    {
        ForkLock_.ReleaseWriter();
        IterateAtForkHandlerSets([] (const TAtForkHandlerSet& set) {
            if (set.Child) {
                set.Child();
            }
        });
    }

    template <class F>
    void IterateAtForkHandlerSets(F func)
    {
        for (const auto& set : AtForkHandlerSets_) {
            if (set.Initialized.load()) {
                func(set);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

static void* AtForkManagerInitializer = [] { return TAtForkManager::Get(); }();

////////////////////////////////////////////////////////////////////////////////

void RegisterAtForkHandlers(
    TAtForkHandler prepare,
    TAtForkHandler parent,
    TAtForkHandler child)
{
    return TAtForkManager::Get()->RegisterAtForkHandlers(
        std::move(prepare),
        std::move(parent),
        std::move(child));
}

TReaderWriterSpinLock* GetForkLock()
{
    return TAtForkManager::Get()->GetForkLock();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

