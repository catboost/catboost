#pragma once

#if !defined(INCLUDE_FUTURE_INL_H)
#error "you should never include wait-inl.h directly"
#endif // INCLUDE_FUTURE_INL_H

namespace NThreading {
    namespace NImpl {
        template <typename TContainer>
        TVector<TFuture<void>> ToVoidFutures(const TContainer& futures) {
            TVector<TFuture<void>> voidFutures;
            voidFutures.reserve(futures.size());

            for (const auto& future: futures) {
                voidFutures.push_back(future.IgnoreResult());
            }

            return voidFutures;
        }
    }

    template <typename TContainer>
    [[nodiscard]] NImpl::EnableGenericWait<TContainer> WaitAll(const TContainer& futures) {
        return WaitAll(NImpl::ToVoidFutures(futures));
    }

    template <typename TContainer>
    [[nodiscard]] NImpl::EnableGenericWait<TContainer> WaitExceptionOrAll(const TContainer& futures) {
        return WaitExceptionOrAll(NImpl::ToVoidFutures(futures));
    }

    template <typename TContainer>
    [[nodiscard]] NImpl::EnableGenericWait<TContainer> WaitAny(const TContainer& futures) {
        return WaitAny(NImpl::ToVoidFutures(futures));
    }
}
