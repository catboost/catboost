#pragma once

#if !defined(INCLUDE_FUTURE_INL_H)
#error "you should never include wait-inl.h directly"
#endif // INCLUDE_FUTURE_INL_H

namespace NThreading {
    namespace NImpl {
        ////////////////////////////////////////////////////////////////////////////////

        struct TWaitExceptionOrAll: public TAtomicRefCount<TWaitExceptionOrAll> {
            TPromise<void> Promise;
            size_t Count;
            TSpinLock Lock;

            TWaitExceptionOrAll(size_t count)
                : Promise(NewPromise())
                , Count(count)
            {
            }

            template<class T>
            void Set(const TFuture<T>& future) {
                TGuard<TSpinLock> guard(Lock);
                try {
                    future.TryRethrow();
                    if (--Count == 0) {
                        Promise.SetValue();
                    }
                } catch (...) {
                    Y_ASSERT(!Promise.HasValue());
                    if (!Promise.HasException()) {
                        Promise.SetException(std::current_exception());
                    }
                }
            }
        };

        ////////////////////////////////////////////////////////////////////////////////

        struct TWaitAll: public TAtomicRefCount<TWaitAll> {
            TPromise<void> Promise;
            size_t Count;
            TSpinLock Lock;
            std::exception_ptr Exception;

            TWaitAll(size_t count)
                : Promise(NewPromise())
                , Count(count)
                , Exception()
            {
            }

            template<class T>
            void Set(const TFuture<T>& future) {
                TGuard<TSpinLock> guard(Lock);

                if (!Exception) {
                    try {
                        future.TryRethrow();
                    } catch (...) {
                        Exception = std::current_exception();
                    }
                }

                if (--Count == 0) {
                    Y_ASSERT(!Promise.HasValue() && !Promise.HasException());
                    if (Exception) {
                        Promise.SetException(std::move(Exception));
                    } else {
                        Promise.SetValue();
                    }
                }
            }
        };

        ////////////////////////////////////////////////////////////////////////////////

        struct TWaitAny: public TAtomicRefCount<TWaitAny> {
            TPromise<void> Promise;
            TSpinLock Lock;

            TWaitAny()
                : Promise(NewPromise())
            {
            }

            template<class T>
            void Set(const TFuture<T>& future) {
                if (Lock.TryAcquire()) {
                    try {
                        future.TryRethrow();
                    } catch (...) {
                        Y_ASSERT(!Promise.HasValue() && !Promise.HasException());
                        Promise.SetException(std::current_exception());
                        return;
                    }
                    Promise.SetValue();
                }
            }
        };

    }

    ////////////////////////////////////////////////////////////////////////////////

    inline TFuture<void> WaitAll(const TFuture<void>& f1) {
        return f1;
    }

    inline TFuture<void> WaitAll(const TFuture<void>& f1, const TFuture<void>& f2) {
        using TCallback = NImpl::TCallback<void>;

        TIntrusivePtr<NImpl::TWaitAll> waiter = new NImpl::TWaitAll(2);
        auto callback = TCallback([=](const TFuture<void>& future) mutable {
            waiter->Set(future);
        });

        f1.Subscribe(callback);
        f2.Subscribe(callback);

        return waiter->Promise;
    }

    template <typename TContainer>
    inline TFuture<void> WaitAll(const TContainer& futures) {
        if (futures.empty()) {
            return MakeFuture();
        }
        if (futures.size() == 1) {
            return futures.front().IgnoreResult();
        }

        using TCallback = NImpl::TCallback<typename TContainer::value_type::value_type>;

        TIntrusivePtr<NImpl::TWaitAll> waiter = new NImpl::TWaitAll(futures.size());
        auto callback = TCallback([=](const auto& future) mutable {
            waiter->Set(future);
        });

        for (auto& fut : futures) {
            fut.Subscribe(callback);
        }

        return waiter->Promise;
    }


    ////////////////////////////////////////////////////////////////////////////////

    inline TFuture<void> WaitExceptionOrAll(const TFuture<void>& f1) {
        return f1;
    }

    inline TFuture<void> WaitExceptionOrAll(const TFuture<void>& f1, const TFuture<void>& f2) {
        using TCallback = NImpl::TCallback<void>;

        TIntrusivePtr<NImpl::TWaitExceptionOrAll> waiter = new NImpl::TWaitExceptionOrAll(2);
        auto callback = TCallback([=](const TFuture<void>& future) mutable {
            waiter->Set(future);
        });

        f1.Subscribe(callback);
        f2.Subscribe(callback);

        return waiter->Promise;
    }

    template <typename TContainer>
    inline TFuture<void> WaitExceptionOrAll(const TContainer& futures) {
        if (futures.empty()) {
            return MakeFuture();
        }
        if (futures.size() == 1) {
            return futures.front().IgnoreResult();
        }

        using TCallback = NImpl::TCallback<typename TContainer::value_type::value_type>;

        TIntrusivePtr<NImpl::TWaitExceptionOrAll> waiter = new NImpl::TWaitExceptionOrAll(futures.size());
        auto callback = TCallback([=](const auto& future) mutable {
            waiter->Set(future);
        });

        for (auto& fut : futures) {
            fut.Subscribe(callback);
        }

        return waiter->Promise;
    }

    ////////////////////////////////////////////////////////////////////////////////

    inline TFuture<void> WaitAny(const TFuture<void>& f1) {
        return f1;
    }

    inline TFuture<void> WaitAny(const TFuture<void>& f1, const TFuture<void>& f2) {
        using TCallback = NImpl::TCallback<void>;

        TIntrusivePtr<NImpl::TWaitAny> waiter = new NImpl::TWaitAny();
        auto callback = TCallback([=](const TFuture<void>& future) mutable {
            waiter->Set(future);
        });

        f1.Subscribe(callback);
        f2.Subscribe(callback);

        return waiter->Promise;
    }

    template <typename TContainer>
    inline TFuture<void> WaitAny(const TContainer& futures) {
        if (futures.empty()) {
            return MakeFuture();
        }

        if (futures.size() == 1) {
            return futures.front().IgnoreResult();
        }

        using TCallback = NImpl::TCallback<typename TContainer::value_type::value_type>;

        TIntrusivePtr<NImpl::TWaitAny> waiter = new NImpl::TWaitAny();
        auto callback = TCallback([=](const auto& future) mutable {
            waiter->Set(future);
        });

        for (auto& fut : futures) {
            fut.Subscribe(callback);
        }

        return waiter->Promise;
    }

}
