#pragma once

#if !defined(INCLUDE_FUTURE_INL_H)
#error "you should never include future-inl.h directly"
#endif // INCLUDE_FUTURE_INL_H

namespace NThreading {
    namespace NImpl {
        ////////////////////////////////////////////////////////////////////////////////

        template <typename T>
        using TCallback = std::function<void(const TFuture<T>&)>;

        template <typename T>
        using TCallbackList = TVector<TCallback<T>>; // TODO: small vector

        ////////////////////////////////////////////////////////////////////////////////

        enum class TError {
            Error
        };

        template <typename T>
        class TFutureState: public TAtomicRefCount<TFutureState<T>> {
            enum {
                NotReady,
                ExceptionSet,
                ValueMoved, // keep the ordering of this and following values
                ValueSet,
                ValueRead,
            };

        private:
            mutable TAtomic State;
            TAdaptiveLock StateLock;

            TCallbackList<T> Callbacks;
            mutable THolder<TSystemEvent> ReadyEvent;

            std::exception_ptr Exception;

            union {
                char NullValue;
                T Value;
            };

            void AccessValue(TDuration timeout, int acquireState) const {
                int state = AtomicGet(State);
                if (Y_UNLIKELY(state == NotReady)) {
                    if (timeout == TDuration::Zero()) {
                        ythrow TFutureException() << "value not set";
                    }

                    if (!Wait(timeout)) {
                        ythrow TFutureException() << "wait timeout";
                    }

                    state = AtomicGet(State);
                }

                TryRethrowWithState(state);

                switch (AtomicGetAndCas(&State, acquireState, ValueSet)) {
                    case ValueSet:
                        break;
                    case ValueRead:
                        if (acquireState != ValueRead) {
                            ythrow TFutureException() << "value being read";
                        }
                        break;
                    case ValueMoved:
                        ythrow TFutureException() << "value was moved";
                    default:
                        Y_ASSERT(state == ValueSet);
                }
            }

        public:
            TFutureState()
                : State(NotReady)
                , NullValue(0)
            {
            }

            template <typename TT>
            TFutureState(TT&& value)
                : State(ValueSet)
                , Value(std::forward<TT>(value))
            {
            }

            TFutureState(std::exception_ptr exception, TError)
                : State(ExceptionSet)
                , Exception(std::move(exception))
                , NullValue(0)
            {
            }

            ~TFutureState() {
                if (State >= ValueMoved) { // ValueMoved, ValueSet, ValueRead
                    Value.~T();
                }
            }

            bool HasValue() const {
                return AtomicGet(State) >= ValueMoved; // ValueMoved, ValueSet, ValueRead
            }

            void TryRethrow() const {
                int state = AtomicGet(State);
                TryRethrowWithState(state);
            }

            bool HasException() const {
                return AtomicGet(State) == ExceptionSet;
            }

            const T& GetValue(TDuration timeout = TDuration::Zero()) const {
                AccessValue(timeout, ValueRead);
                return Value;
            }

            T ExtractValue(TDuration timeout = TDuration::Zero()) {
                AccessValue(timeout, ValueMoved);
                return std::move(Value);
            }

            template <typename TT>
            void SetValue(TT&& value) {
                bool success = TrySetValue(std::forward<TT>(value));
                if (Y_UNLIKELY(!success)) {
                    ythrow TFutureException() << "value already set";
                }
            }

            template <typename TT>
            bool TrySetValue(TT&& value) {
                TSystemEvent* readyEvent = nullptr;
                TCallbackList<T> callbacks;

                with_lock (StateLock) {
                    int state = AtomicGet(State);
                    if (Y_UNLIKELY(state != NotReady)) {
                        return false;
                    }

                    new (&Value) T(std::forward<TT>(value));

                    readyEvent = ReadyEvent.Get();
                    callbacks = std::move(Callbacks);

                    AtomicSet(State, ValueSet);
                }

                if (readyEvent) {
                    readyEvent->Signal();
                }

                if (callbacks) {
                    TFuture<T> temp(this);
                    for (auto& callback : callbacks) {
                        callback(temp);
                    }
                }

                return true;
            }

            void SetException(std::exception_ptr e) {
                bool success = TrySetException(std::move(e));
                if (Y_UNLIKELY(!success)) {
                    ythrow TFutureException() << "value already set";
                }
            }

            bool TrySetException(std::exception_ptr e) {
                TSystemEvent* readyEvent;
                TCallbackList<T> callbacks;

                with_lock (StateLock) {
                    int state = AtomicGet(State);
                    if (Y_UNLIKELY(state != NotReady)) {
                        return false;
                    }

                    Exception = std::move(e);

                    readyEvent = ReadyEvent.Get();
                    callbacks = std::move(Callbacks);

                    AtomicSet(State, ExceptionSet);
                }

                if (readyEvent) {
                    readyEvent->Signal();
                }

                if (callbacks) {
                    TFuture<T> temp(this);
                    for (auto& callback : callbacks) {
                        callback(temp);
                    }
                }

                return true;
            }

            template <typename F>
            bool Subscribe(F&& func) {
                with_lock (StateLock) {
                    int state = AtomicGet(State);
                    if (state == NotReady) {
                        Callbacks.emplace_back(std::forward<F>(func));
                        return true;
                    }
                }
                return false;
            }

            void Wait() const {
                Wait(TInstant::Max());
            }

            bool Wait(TDuration timeout) const {
                return Wait(timeout.ToDeadLine());
            }

            bool Wait(TInstant deadline) const {
                TSystemEvent* readyEvent = nullptr;

                with_lock (StateLock) {
                    int state = AtomicGet(State);
                    if (state != NotReady) {
                        return true;
                    }

                    if (!ReadyEvent) {
                        ReadyEvent.Reset(new TSystemEvent());
                    }
                    readyEvent = ReadyEvent.Get();
                }

                Y_ASSERT(readyEvent);
                return readyEvent->WaitD(deadline);
            }

            void TryRethrowWithState(int state) const {
                if (Y_UNLIKELY(state == ExceptionSet)) {
                    Y_ASSERT(Exception);
                    std::rethrow_exception(Exception);
                }
            }
        };

        ////////////////////////////////////////////////////////////////////////////////

        template <>
        class TFutureState<void>: public TAtomicRefCount<TFutureState<void>> {
            enum {
                NotReady,
                ValueSet,
                ExceptionSet,
            };

        private:
            TAtomic State;
            TAdaptiveLock StateLock;

            TCallbackList<void> Callbacks;
            mutable THolder<TSystemEvent> ReadyEvent;

            std::exception_ptr Exception;

        public:
            TFutureState(bool valueSet = false)
                : State(valueSet ? ValueSet : NotReady)
            {
            }

            TFutureState(std::exception_ptr exception, TError)
                : State(ExceptionSet)
                , Exception(std::move(exception))
            {
            }

            bool HasValue() const {
                return AtomicGet(State) == ValueSet;
            }

            void TryRethrow() const {
                int state = AtomicGet(State);
                TryRethrowWithState(state);
            }

            bool HasException() const {
                return AtomicGet(State) == ExceptionSet;
            }

            void GetValue(TDuration timeout = TDuration::Zero()) const {
                int state = AtomicGet(State);
                if (Y_UNLIKELY(state == NotReady)) {
                    if (timeout == TDuration::Zero()) {
                        ythrow TFutureException() << "value not set";
                    }

                    if (!Wait(timeout)) {
                        ythrow TFutureException() << "wait timeout";
                    }

                    state = AtomicGet(State);
                }

                TryRethrowWithState(state);

                Y_ASSERT(state == ValueSet);
            }

            void SetValue() {
                bool success = TrySetValue();
                if (Y_UNLIKELY(!success)) {
                    ythrow TFutureException() << "value already set";
                }
            }

            bool TrySetValue() {
                TSystemEvent* readyEvent = nullptr;
                TCallbackList<void> callbacks;

                with_lock (StateLock) {
                    int state = AtomicGet(State);
                    if (Y_UNLIKELY(state != NotReady)) {
                        return false;
                    }

                    readyEvent = ReadyEvent.Get();
                    callbacks = std::move(Callbacks);

                    AtomicSet(State, ValueSet);
                }

                if (readyEvent) {
                    readyEvent->Signal();
                }

                if (callbacks) {
                    TFuture<void> temp(this);
                    for (auto& callback : callbacks) {
                        callback(temp);
                    }
                }

                return true;
            }

            void SetException(std::exception_ptr e) {
                bool success = TrySetException(std::move(e));
                if (Y_UNLIKELY(!success)) {
                    ythrow TFutureException() << "value already set";
                }
            }

            bool TrySetException(std::exception_ptr e) {
                TSystemEvent* readyEvent = nullptr;
                TCallbackList<void> callbacks;

                with_lock (StateLock) {
                    int state = AtomicGet(State);
                    if (Y_UNLIKELY(state != NotReady)) {
                        return false;
                    }

                    Exception = std::move(e);

                    readyEvent = ReadyEvent.Get();
                    callbacks = std::move(Callbacks);

                    AtomicSet(State, ExceptionSet);
                }

                if (readyEvent) {
                    readyEvent->Signal();
                }

                if (callbacks) {
                    TFuture<void> temp(this);
                    for (auto& callback : callbacks) {
                        callback(temp);
                    }
                }

                return true;
            }

            template <typename F>
            bool Subscribe(F&& func) {
                with_lock (StateLock) {
                    int state = AtomicGet(State);
                    if (state == NotReady) {
                        Callbacks.emplace_back(std::forward<F>(func));
                        return true;
                    }
                }
                return false;
            }

            void Wait() const {
                Wait(TInstant::Max());
            }

            bool Wait(TDuration timeout) const {
                return Wait(timeout.ToDeadLine());
            }

            bool Wait(TInstant deadline) const {
                TSystemEvent* readyEvent = nullptr;

                with_lock (StateLock) {
                    int state = AtomicGet(State);
                    if (state != NotReady) {
                        return true;
                    }

                    if (!ReadyEvent) {
                        ReadyEvent.Reset(new TSystemEvent());
                    }
                    readyEvent = ReadyEvent.Get();
                }

                Y_ASSERT(readyEvent);
                return readyEvent->WaitD(deadline);
            }

            void TryRethrowWithState(int state) const {
                if (Y_UNLIKELY(state == ExceptionSet)) {
                    Y_ASSERT(Exception);
                    std::rethrow_exception(Exception);
                }
            }
        };

        ////////////////////////////////////////////////////////////////////////////////

        template <typename T>
        inline void SetValueImpl(TPromise<T>& promise, const T& value) {
            promise.SetValue(value);
        }

        template <typename T>
        inline void SetValueImpl(TPromise<T>& promise, T&& value) {
            promise.SetValue(std::move(value));
        }

        template <typename T>
        inline void SetValueImpl(TPromise<T>& promise, const TFuture<T>& future) {
            future.Subscribe([=](const TFuture<T>& f) mutable {
                try {
                    promise.SetValue(f.GetValue());
                } catch (...) {
                    promise.SetException(std::current_exception());
                }
            });
        }

        inline void SetValueImpl(TPromise<void>& promise, const TFuture<void>& future) {
            future.Subscribe([=](const TFuture<void>& f) mutable {
                try {
                    f.TryRethrow();
                    promise.SetValue();
                } catch (...) {
                    promise.SetException(std::current_exception());
                }
            });
        }

        template <typename T, typename F>
        inline void SetValue(TPromise<T>& promise, F&& func) {
            try {
                SetValueImpl(promise, func());
            } catch (...) {
                promise.SetException(std::current_exception());
            }
        }

        template <typename F>
        inline void SetValue(TPromise<void>& promise, F&& func,
                             std::enable_if_t<std::is_void<TFunctionResult<F>>::value, bool> = false) {
            try {
                func();
                promise.SetValue();
            } catch (...) {
                promise.SetException(std::current_exception());
            }
        }

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
                        Promise.SetValue();
                    } catch (...) {
                        Y_ASSERT(!Promise.HasValue() && !Promise.HasException());
                        Promise.SetException(std::current_exception());
                    }
                }
            }
        };

    }

    ////////////////////////////////////////////////////////////////////////////////

    class TFutureStateId {
    private:
        const void* Id;

    public:
        template <typename T>
        explicit TFutureStateId(const NImpl::TFutureState<T>& state)
            : Id(&state)
        {
        }

        const void* Value() const noexcept {
            return Id;
        }
    };

    inline bool operator==(const TFutureStateId& l, const TFutureStateId& r) {
        return l.Value() == r.Value();
    }

    inline bool operator!=(const TFutureStateId& l, const TFutureStateId& r) {
        return !(l == r);
    }

    ////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline TFuture<T>::TFuture(const TIntrusivePtr<TFutureState>& state) noexcept
        : State(state)
    {
    }

    template <typename T>
    inline void TFuture<T>::Swap(TFuture<T>& other) {
        State.Swap(other.State);
    }

    template <typename T>
    inline bool TFuture<T>::HasValue() const {
        return State && State->HasValue();
    }

    template <typename T>
    inline const T& TFuture<T>::GetValue(TDuration timeout) const {
        EnsureInitialized();
        return State->GetValue(timeout);
    }

    template <typename T>
    inline T TFuture<T>::ExtractValue(TDuration timeout) {
        EnsureInitialized();
        return State->ExtractValue(timeout);
    }

    template <typename T>
    inline const T& TFuture<T>::GetValueSync() const {
        return GetValue(TDuration::Max());
    }

    template <typename T>
    inline T TFuture<T>::ExtractValueSync() {
        return ExtractValue(TDuration::Max());
    }

    template <typename T>
    inline void TFuture<T>::TryRethrow() const {
        if (State) {
            State->TryRethrow();
        }
    }

    template <typename T>
    inline bool TFuture<T>::HasException() const {
        return State && State->HasException();
    }

    template <typename T>
    inline void TFuture<T>::Wait() const {
        EnsureInitialized();
        return State->Wait();
    }

    template <typename T>
    inline bool TFuture<T>::Wait(TDuration timeout) const {
        EnsureInitialized();
        return State->Wait(timeout);
    }

    template <typename T>
    inline bool TFuture<T>::Wait(TInstant deadline) const {
        EnsureInitialized();
        return State->Wait(deadline);
    }

    template <typename T>
    template <typename F>
    inline const TFuture<T>& TFuture<T>::Subscribe(F&& func) const {
        EnsureInitialized();
        if (!State->Subscribe(std::forward<F>(func))) {
            func(*this);
        }
        return *this;
    }

    template <typename T>
    template <typename F>
    inline TFuture<TFutureType<TFutureCallResult<F, T>>> TFuture<T>::Apply(F&& func) const {
        auto promise = NewPromise<TFutureType<TFutureCallResult<F, T>>>();
        Subscribe([=](const TFuture<T>& future) mutable {
            NImpl::SetValue(promise, [&]() mutable { return func(future); });
        });
        return promise;
    }

    template <typename T>
    inline TFuture<void> TFuture<T>::IgnoreResult() const {
        auto promise = NewPromise();
        Subscribe([=](const TFuture<T>& future) mutable {
            try {
                future.TryRethrow();
                promise.SetValue();
            } catch (...) {
                promise.SetException(std::current_exception());
            }
        });
        return promise;
    }

    template <typename T>
    inline bool TFuture<T>::Initialized() const {
        return bool(State);
    }

    template <typename T>
    inline TMaybe<TFutureStateId> TFuture<T>::StateId() const noexcept {
        return State != nullptr ? MakeMaybe<TFutureStateId>(*State) : Nothing();
    }

    template <typename T>
    inline void TFuture<T>::EnsureInitialized() const {
        if (!State) {
            ythrow TFutureException() << "state not initialized";
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

    inline TFuture<void>::TFuture(const TIntrusivePtr<TFutureState>& state) noexcept
        : State(state)
    {
    }

    inline void TFuture<void>::Swap(TFuture<void>& other) {
        State.Swap(other.State);
    }

    inline bool TFuture<void>::HasValue() const {
        return State && State->HasValue();
    }

    inline void TFuture<void>::GetValue(TDuration timeout) const {
        EnsureInitialized();
        State->GetValue(timeout);
    }

    inline void TFuture<void>::GetValueSync() const {
        GetValue(TDuration::Max());
    }

    inline void TFuture<void>::TryRethrow() const {
        if (State) {
            State->TryRethrow();
        }
    }

    inline bool TFuture<void>::HasException() const {
        return State && State->HasException();
    }

    inline void TFuture<void>::Wait() const {
        EnsureInitialized();
        return State->Wait();
    }

    inline bool TFuture<void>::Wait(TDuration timeout) const {
        EnsureInitialized();
        return State->Wait(timeout);
    }

    inline bool TFuture<void>::Wait(TInstant deadline) const {
        EnsureInitialized();
        return State->Wait(deadline);
    }

    template <typename F>
    inline const TFuture<void>& TFuture<void>::Subscribe(F&& func) const {
        EnsureInitialized();
        if (!State->Subscribe(std::forward<F>(func))) {
            func(*this);
        }
        return *this;
    }

    template <typename F>
    inline TFuture<TFutureType<TFutureCallResult<F, void>>> TFuture<void>::Apply(F&& func) const {
        auto promise = NewPromise<TFutureType<TFutureCallResult<F, void>>>();
        Subscribe([=](const TFuture<void>& future) mutable {
            NImpl::SetValue(promise, [&]() mutable { return func(future); });
        });
        return promise;
    }

    template <typename R>
    inline TFuture<R> TFuture<void>::Return(const R& value) const {
        auto promise = NewPromise<R>();
        Subscribe([=](const TFuture<void>& future) mutable {
            try {
                future.TryRethrow();
                promise.SetValue(value);
            } catch (...) {
                promise.SetException(std::current_exception());
            }
        });
        return promise;
    }

    inline bool TFuture<void>::Initialized() const {
        return bool(State);
    }

    inline TMaybe<TFutureStateId> TFuture<void>::StateId() const noexcept {
        return State != nullptr ? MakeMaybe<TFutureStateId>(*State) : Nothing();
    }

    inline void TFuture<void>::EnsureInitialized() const {
        if (!State) {
            ythrow TFutureException() << "state not initialized";
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline TPromise<T>::TPromise(const TIntrusivePtr<TFutureState>& state) noexcept
        : State(state)
    {
    }

    template <typename T>
    inline void TPromise<T>::Swap(TPromise<T>& other) {
        State.Swap(other.State);
    }

    template <typename T>
    inline const T& TPromise<T>::GetValue() const {
        EnsureInitialized();
        return State->GetValue();
    }

    template <typename T>
    inline T TPromise<T>::ExtractValue() {
        EnsureInitialized();
        return State->ExtractValue();
    }

    template <typename T>
    inline bool TPromise<T>::HasValue() const {
        return State && State->HasValue();
    }

    template <typename T>
    inline void TPromise<T>::SetValue(const T& value) {
        EnsureInitialized();
        State->SetValue(value);
    }

    template <typename T>
    inline void TPromise<T>::SetValue(T&& value) {
        EnsureInitialized();
        State->SetValue(std::move(value));
    }

    template <typename T>
    inline bool TPromise<T>::TrySetValue(const T& value) {
        EnsureInitialized();
        return State->TrySetValue(value);
    }

    template <typename T>
    inline bool TPromise<T>::TrySetValue(T&& value) {
        EnsureInitialized();
        return State->TrySetValue(std::move(value));
    }

    template <typename T>
    inline void TPromise<T>::TryRethrow() const {
        if (State) {
            State->TryRethrow();
        }
    }

    template <typename T>
    inline bool TPromise<T>::HasException() const {
        return State && State->HasException();
    }

    template <typename T>
    inline void TPromise<T>::SetException(const TString& e) {
        EnsureInitialized();
        State->SetException(std::make_exception_ptr(yexception() << e));
    }

    template <typename T>
    inline void TPromise<T>::SetException(std::exception_ptr e) {
        EnsureInitialized();
        State->SetException(std::move(e));
    }

    template <typename T>
    inline bool TPromise<T>::TrySetException(std::exception_ptr e) {
        EnsureInitialized();
        return State->TrySetException(std::move(e));
    }

    template <typename T>
    inline TFuture<T> TPromise<T>::GetFuture() const {
        EnsureInitialized();
        return TFuture<T>(State);
    }

    template <typename T>
    inline TPromise<T>::operator TFuture<T>() const {
        return GetFuture();
    }

    template <typename T>
    inline bool TPromise<T>::Initialized() const {
        return bool(State);
    }

    template <typename T>
    inline void TPromise<T>::EnsureInitialized() const {
        if (!State) {
            ythrow TFutureException() << "state not initialized";
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

    inline TPromise<void>::TPromise(const TIntrusivePtr<TFutureState>& state) noexcept
        : State(state)
    {
    }

    inline void TPromise<void>::Swap(TPromise<void>& other) {
        State.Swap(other.State);
    }

    inline void TPromise<void>::GetValue() const {
        EnsureInitialized();
        State->GetValue();
    }

    inline bool TPromise<void>::HasValue() const {
        return State && State->HasValue();
    }

    inline void TPromise<void>::SetValue() {
        EnsureInitialized();
        State->SetValue();
    }

    inline bool TPromise<void>::TrySetValue() {
        EnsureInitialized();
        return State->TrySetValue();
    }

    inline void TPromise<void>::TryRethrow() const {
        if(State) {
            State->TryRethrow();
        }
    }

    inline bool TPromise<void>::HasException() const {
        return State && State->HasException();
    }

    inline void TPromise<void>::SetException(const TString& e) {
        EnsureInitialized();
        State->SetException(std::make_exception_ptr(yexception() << e));
    }

    inline void TPromise<void>::SetException(std::exception_ptr e) {
        EnsureInitialized();
        State->SetException(std::move(e));
    }

    inline bool TPromise<void>::TrySetException(std::exception_ptr e) {
        EnsureInitialized();
        return State->TrySetException(std::move(e));
    }

    inline TFuture<void> TPromise<void>::GetFuture() const {
        EnsureInitialized();
        return TFuture<void>(State);
    }

    inline TPromise<void>::operator TFuture<void>() const {
        return GetFuture();
    }

    inline bool TPromise<void>::Initialized() const {
        return bool(State);
    }

    inline void TPromise<void>::EnsureInitialized() const {
        if (!State) {
            ythrow TFutureException() << "state not initialized";
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline TPromise<T> NewPromise() {
        return {new NImpl::TFutureState<T>()};
    }

    inline TPromise<void> NewPromise() {
        return {new NImpl::TFutureState<void>()};
    }

    template <typename T>
    inline TFuture<T> MakeFuture(const T& value) {
        return {new NImpl::TFutureState<T>(value)};
    }

    template <typename T>
    inline TFuture<std::remove_reference_t<T>> MakeFuture(T&& value) {
        return {new NImpl::TFutureState<std::remove_reference_t<T>>(std::forward<T>(value))};
    }

    template <typename T>
    inline TFuture<T> MakeFuture() {
        struct TCache {
            TFuture<T> Instance{new NImpl::TFutureState<T>(Default<T>())};
        };
        return Singleton<TCache>()->Instance;
    }

    template <typename T>
    inline TFuture<T> MakeErrorFuture(std::exception_ptr exception)
    {
        return {new NImpl::TFutureState<T>(std::move(exception), NImpl::TError::Error)};
    }

    inline TFuture<void> MakeFuture() {
        struct TCache {
            TFuture<void> Instance{new NImpl::TFutureState<void>(true)};
        };
        return Singleton<TCache>()->Instance;
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
