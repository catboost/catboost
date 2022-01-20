#pragma once

#if !defined(INCLUDE_FUTURE_INL_H)
#error "you should never include wait_group-inl.h directly"
#endif // INCLUDE_FUTURE_INL_H

#include "wait_policy.h"

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>

#include <library/cpp/threading/future/core/future.h>

#include <util/system/spinlock.h>

#include <atomic>
#include <exception>

namespace NThreading {
    namespace NWaitGroup::NImpl {
        template <class WaitPolicy>
        struct TState final : TAtomicRefCount<TState<WaitPolicy>> {
            template <class T>
            void Add(const TFuture<T>& future);
            TFuture<void> Finish();

            void TryPublish();
            void Publish();

            bool ShouldPublishByCount() const noexcept;
            bool ShouldPublishByException() const noexcept;

            TStateRef<WaitPolicy> SharedFromThis() noexcept {
                return TStateRef<WaitPolicy>{this};
            }

            enum class EPhase {
                Initial,
                Publishing,
            };

            // initially we have one imaginary discovered future which we
            // use for synchronization with ::Finish
            std::atomic<ui64> Discovered{1};

            std::atomic<ui64> Finished{0};

            std::atomic<EPhase> Phase{EPhase::Initial};

            TPromise<void> Subscribers = NewPromise();

            mutable TAdaptiveLock Mut;
            std::exception_ptr ExceptionInFlight;

            void TrySetException(std::exception_ptr eptr) noexcept {
                TGuard lock{Mut};
                if (!ExceptionInFlight) {
                    ExceptionInFlight = std::move(eptr);
                }
            }

            std::exception_ptr GetExceptionInFlight() const noexcept {
                TGuard lock{Mut};
                return ExceptionInFlight;
            }
        };

        template <class WaitPolicy>
        inline TFuture<void> TState<WaitPolicy>::Finish() {
            Finished.fetch_add(1); // complete the imaginary future

            // handle empty case explicitly:
            if (Discovered.load() == 1) {
                Y_ASSERT(Phase.load() == EPhase::Initial);
                Publish();
            } else {
                TryPublish();
            }

            return Subscribers;
        }

        template <class WaitPolicy>
        template <class T>
        inline void TState<WaitPolicy>::Add(const TFuture<T>& future) {
            future.EnsureInitialized();

            Discovered.fetch_add(1);

            // NoexceptSubscribe is needed to make ::Add exception-safe
            future.NoexceptSubscribe([self = SharedFromThis()](auto&& future) {
                try {
                    future.TryRethrow();
                } catch (...) {
                    self->TrySetException(std::current_exception());
                }

                self->Finished.fetch_add(1);
                self->TryPublish();
            });
        }

        //
        // ============================ PublishByCount ==================================
        //

        template <class WaitPolicy>
        inline bool TState<WaitPolicy>::ShouldPublishByCount() const noexcept {
            // - safety: a) If the future incremented ::Finished, and we observe the effect, then we will observe ::Discovered as incremented by its discovery later
            //           b) Every discovery of a future observes discovery of the imaginary future
            //          a, b => if finishedByNow == discoveredByNow, then every future discovered in [imaginary discovered, imaginary finished] is finished
            //
            // - liveness: a) TryPublish is called after each increment of ::Finished
            //             b) There is some last increment of ::Finished which follows all other operations with ::Finished and ::Discovered (provided that every future is eventually set)
            //             c) For each increment of ::Discovered there is an increment of ::Finished (provided that every future is eventually set)
            //          a, b c => some call to ShouldPublishByCount will always return true
            //
            // order of the following two operations is significant for the proof.
            auto finishedByNow = Finished.load();
            auto discoveredByNow = Discovered.load();

            return finishedByNow == discoveredByNow;
        }

        template <>
        inline bool TState<TWaitPolicy::TAny>::ShouldPublishByCount() const noexcept {
            auto finishedByNow = Finished.load();

            // note that the empty case is not handled here
            return finishedByNow >= 2; // at least one non-imaginary
        }

        //
        // ============================ PublishByException ==================================
        //

        template <>
        inline bool TState<TWaitPolicy::TAny>::ShouldPublishByException() const noexcept {
            // for TAny exceptions are handled by ShouldPublishByCount
            return false;
        }

        template <>
        inline bool TState<TWaitPolicy::TAll>::ShouldPublishByException() const noexcept {
            return false;
        }

        template <>
        inline bool TState<TWaitPolicy::TExceptionOrAll>::ShouldPublishByException() const noexcept {
            return GetExceptionInFlight() != nullptr;
        }

        //
        //
        //

        template <class WaitPolicy>
        inline void TState<WaitPolicy>::TryPublish() {
            // the order is insignificant (without proof)
            bool shouldPublish = ShouldPublishByCount() || ShouldPublishByException();

            if (shouldPublish) {
                if (auto currentPhase = EPhase::Initial;
                    Phase.compare_exchange_strong(currentPhase, EPhase::Publishing)) {
                    Publish();
                }
            }
        }

        template <class WaitPolicy>
        inline void TState<WaitPolicy>::Publish() {
            auto eptr = GetExceptionInFlight();

            // can potentially throw
            if (eptr) {
                Subscribers.SetException(std::move(eptr));
            } else {
                Subscribers.SetValue();
            }
        }
    }

    template <class WaitPolicy>
    inline TWaitGroup<WaitPolicy>::TWaitGroup()
        : State_{MakeIntrusive<NWaitGroup::NImpl::TState<WaitPolicy>>()}
    {
    }

    template <class WaitPolicy>
    template <class T>
    inline TWaitGroup<WaitPolicy>& TWaitGroup<WaitPolicy>::Add(const TFuture<T>& future) {
        State_->Add(future);
        return *this;
    }

    template <class WaitPolicy>
    inline TFuture<void> TWaitGroup<WaitPolicy>::Finish() && {
        auto res = State_->Finish();

        // just to prevent nasty bugs from use-after-move
        State_.Reset();

        return res;
    }
}

