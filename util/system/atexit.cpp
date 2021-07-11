#include "atexit.h"
#include "atomic.h"
#include "yassert.h"
#include "spinlock.h"
#include "thread.h"

#include <util/generic/ylimits.h>
#include <util/generic/utility.h>
#include <util/generic/deque.h>
#include <util/generic/queue.h>

#include <tuple>

#include <cstdlib>

namespace {
    class TAtExit {
        struct TFunc {
            TAtExitFunc Func;
            void* Ctx;
            size_t Priority;
            size_t Number;
        };

        struct TCmp {
            inline bool operator()(const TFunc* l, const TFunc* r) const noexcept {
                return std::tie(l->Priority, l->Number) < std::tie(r->Priority, r->Number);
            }
        };

    public:
        inline TAtExit() noexcept
            : FinishStarted_(0)
        {
        }

        inline void Finish() noexcept {
            AtomicSet(FinishStarted_, 1);

            auto guard = Guard(Lock_);

            while (Items_) {
                auto c = Items_.top();

                Y_ASSERT(c);

                Items_.pop();

                {
                    auto unguard = Unguard(guard);

                    try {
                        c->Func(c->Ctx);
                    } catch (...) {
                        // ¯\_(ツ)_/¯
                    }
                }
            }
        }

        inline void Register(TAtExitFunc func, void* ctx, size_t priority) {
            with_lock (Lock_) {
                Store_.push_back({func, ctx, priority, Store_.size()});
                Items_.push(&Store_.back());
            }
        }

        inline bool FinishStarted() const {
            return AtomicGet(FinishStarted_);
        }

    private:
        TAdaptiveLock Lock_;
        TAtomic FinishStarted_;
        TDeque<TFunc> Store_;
        TPriorityQueue<TFunc*, TVector<TFunc*>, TCmp> Items_;
    };

    static TAtomic atExitLock = 0;
    static TAtExit* volatile atExitPtr = nullptr;
    alignas(TAtExit) static char atExitMem[sizeof(TAtExit)];

    static void OnExit() {
        if (TAtExit* const atExit = AtomicGet(atExitPtr)) {
            atExit->Finish();
            atExit->~TAtExit();
            AtomicSet(atExitPtr, nullptr);
        }
    }

    static inline TAtExit* Instance() {
        if (TAtExit* const atExit = AtomicGet(atExitPtr)) {
            return atExit;
        }
        with_lock (atExitLock) {
            if (TAtExit* const atExit = AtomicGet(atExitPtr)) {
                return atExit;
            }
            atexit(OnExit);
            TAtExit* const atExit = new (atExitMem) TAtExit;
            AtomicSet(atExitPtr, atExit);
            return atExit;
        }
    }
}

void ManualRunAtExitFinalizers() {
    OnExit();
}

bool ExitStarted() {
    if (TAtExit* const atExit = AtomicGet(atExitPtr)) {
        return atExit->FinishStarted();
    }
    return false;
}

void AtExit(TAtExitFunc func, void* ctx, size_t priority) {
    Instance()->Register(func, ctx, priority);
}

void AtExit(TAtExitFunc func, void* ctx) {
    AtExit(func, ctx, Max<size_t>());
}

static void TraditionalCloser(void* ctx) {
    reinterpret_cast<TTraditionalAtExitFunc>(ctx)();
}

void AtExit(TTraditionalAtExitFunc func) {
    AtExit(TraditionalCloser, reinterpret_cast<void*>(func));
}

void AtExit(TTraditionalAtExitFunc func, size_t priority) {
    AtExit(TraditionalCloser, reinterpret_cast<void*>(func), priority);
}
