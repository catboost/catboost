#pragma once

#if !defined(INCLUDE_FUTURE_INL_H)
#error "you should never include wait-inl.h directly"
#endif // INCLUDE_FUTURE_INL_H

#include "wait_group.h"
#include "wait_policy.h"

namespace NThreading {
    namespace NImpl {
        template <class WaitPolicy>
        inline TFuture<void> WaitGeneric(const TFuture<void>& f1) {
            return f1;
        }

        template <class WaitPolicy>
        inline TFuture<void> WaitGeneric(const TFuture<void>& f1, const TFuture<void>& f2) {
            TWaitGroup<WaitPolicy> wg;

            wg.Add(f1).Add(f2);

            return std::move(wg).Finish();
        }

        template <class WaitPolicy, class TContainer>
        inline TFuture<void> WaitGeneric(const TContainer& futures) {
            if (futures.empty()) {
                return MakeFuture();
            }
            if (futures.size() == 1) {
                return futures.front().IgnoreResult();
            }

            TWaitGroup<WaitPolicy> wg;
            for (const auto& fut : futures) {
                wg.Add(fut);
            }

            return std::move(wg).Finish();
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

    inline TFuture<void> WaitAll(const TFuture<void>& f1) {
        return NImpl::WaitGeneric<TWaitPolicy::TAll>(f1);
    }

    inline TFuture<void> WaitAll(const TFuture<void>& f1, const TFuture<void>& f2) {
        return NImpl::WaitGeneric<TWaitPolicy::TAll>(f1, f2);
    }

    template <typename TContainer>
    inline TFuture<void> WaitAll(const TContainer& futures) {
        return NImpl::WaitGeneric<TWaitPolicy::TAll>(futures);
    }


    ////////////////////////////////////////////////////////////////////////////////

    inline TFuture<void> WaitExceptionOrAll(const TFuture<void>& f1) {
        return NImpl::WaitGeneric<TWaitPolicy::TExceptionOrAll>(f1);
    }

    inline TFuture<void> WaitExceptionOrAll(const TFuture<void>& f1, const TFuture<void>& f2) {
        return NImpl::WaitGeneric<TWaitPolicy::TExceptionOrAll>(f1, f2);
    }

    template <typename TContainer>
    inline TFuture<void> WaitExceptionOrAll(const TContainer& futures) {
        return NImpl::WaitGeneric<TWaitPolicy::TExceptionOrAll>(futures);
    }

    ////////////////////////////////////////////////////////////////////////////////

    inline TFuture<void> WaitAny(const TFuture<void>& f1) {
        return NImpl::WaitGeneric<TWaitPolicy::TAny>(f1);
    }

    inline TFuture<void> WaitAny(const TFuture<void>& f1, const TFuture<void>& f2) {
        return NImpl::WaitGeneric<TWaitPolicy::TAny>(f1, f2);
    }

    template <typename TContainer>
    inline TFuture<void> WaitAny(const TContainer& futures) {
        return NImpl::WaitGeneric<TWaitPolicy::TAny>(futures);
    }
}
