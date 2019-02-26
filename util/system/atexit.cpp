#include "atexit.h"
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
    static TAtExit* atExit = nullptr;
    alignas(TAtExit) static char atExitMem[sizeof(TAtExit)];

    static void OnExit() {
        if (atExit) {
            atExit->Finish();
            atExit->~TAtExit();
            atExit = nullptr;
        }
    }

    static inline TAtExit* Instance() {
        if (!atExit) {
            with_lock (atExitLock) {
                if (!atExit) {
                    atexit(OnExit);
                    atExit = new (atExitMem) TAtExit;
                }
            }
        }

        return atExit;
    }
}

bool ExitStarted() {
    if (!atExit) {
        return false;
    }
    return atExit->FinishStarted();
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
