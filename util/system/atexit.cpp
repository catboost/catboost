#include "atexit.h"
#include "yassert.h"
#include "spinlock.h"
#include "thread.h"

#include <util/generic/ylimits.h>
#include <util/generic/utility.h>
#include <util/generic/deque.h>
#include <util/generic/queue.h>

#include <atomic>
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
            : FinishStarted_(false)
        {
        }

        inline void Finish() noexcept {
            FinishStarted_.store(true);

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
            return FinishStarted_.load();
        }

    private:
        TAdaptiveLock Lock_;
        std::atomic<bool> FinishStarted_;
        TDeque<TFunc> Store_;
        TPriorityQueue<TFunc*, TVector<TFunc*>, TCmp> Items_;
    };

    static TAdaptiveLock atExitLock;
    static std::atomic<TAtExit*> atExitPtr = nullptr;
    alignas(TAtExit) static char atExitMem[sizeof(TAtExit)];

    static void OnExit() {
        if (TAtExit* const atExit = atExitPtr.load()) {
            atExit->Finish();
            atExit->~TAtExit();
            atExitPtr.store(nullptr);
        }
    }

    static inline TAtExit* Instance() {
        if (TAtExit* const atExit = atExitPtr.load(std::memory_order_acquire)) {
            return atExit;
        }
        with_lock (atExitLock) {
            if (TAtExit* const atExit = atExitPtr.load()) {
                return atExit;
            }
            atexit(OnExit);
            TAtExit* const atExit = new (atExitMem) TAtExit;
            atExitPtr.store(atExit, std::memory_order_release);
            return atExit;
        }
    }
}

void ManualRunAtExitFinalizers() {
    OnExit();
}

bool ExitStarted() {
    if (TAtExit* const atExit = atExitPtr.load(std::memory_order_acquire)) {
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
