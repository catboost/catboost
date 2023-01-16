#include "wait.h"

#include "wait_group.h"
#include "wait_policy.h"

namespace NThreading {
    namespace {
        template <class WaitPolicy>
        TFuture<void> WaitGeneric(const TFuture<void>& f1) {
            return f1;
        }

        template <class WaitPolicy>
        TFuture<void> WaitGeneric(const TFuture<void>& f1, const TFuture<void>& f2) {
            TWaitGroup<WaitPolicy> wg;

            wg.Add(f1).Add(f2);

            return std::move(wg).Finish();
        }

        template <class WaitPolicy>
        TFuture<void> WaitGeneric(TArrayRef<const TFuture<void>> futures) {
            if (futures.empty()) {
                return MakeFuture();
            }
            if (futures.size() == 1) {
                return futures.front();
            }

            TWaitGroup<WaitPolicy> wg;
            for (const auto& fut : futures) {
                wg.Add(fut);
            }

            return std::move(wg).Finish();
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

    TFuture<void> WaitAll(const TFuture<void>& f1) {
        return WaitGeneric<TWaitPolicy::TAll>(f1);
    }

    TFuture<void> WaitAll(const TFuture<void>& f1, const TFuture<void>& f2) {
        return WaitGeneric<TWaitPolicy::TAll>(f1, f2);
    }

    TFuture<void> WaitAll(TArrayRef<const TFuture<void>> futures) {
        return WaitGeneric<TWaitPolicy::TAll>(futures);
    }


    ////////////////////////////////////////////////////////////////////////////////

    TFuture<void> WaitExceptionOrAll(const TFuture<void>& f1) {
        return WaitGeneric<TWaitPolicy::TExceptionOrAll>(f1);
    }

    TFuture<void> WaitExceptionOrAll(const TFuture<void>& f1, const TFuture<void>& f2) {
        return WaitGeneric<TWaitPolicy::TExceptionOrAll>(f1, f2);
    }

    TFuture<void> WaitExceptionOrAll(TArrayRef<const TFuture<void>> futures) {
        return WaitGeneric<TWaitPolicy::TExceptionOrAll>(futures);
    }

    ////////////////////////////////////////////////////////////////////////////////

    TFuture<void> WaitAny(const TFuture<void>& f1) {
        return WaitGeneric<TWaitPolicy::TAny>(f1);
    }

    TFuture<void> WaitAny(const TFuture<void>& f1, const TFuture<void>& f2) {
        return WaitGeneric<TWaitPolicy::TAny>(f1, f2);
    }

    TFuture<void> WaitAny(TArrayRef<const TFuture<void>> futures) {
        return WaitGeneric<TWaitPolicy::TAny>(futures);
    }
}
