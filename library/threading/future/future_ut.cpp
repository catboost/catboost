#include "future.h"

#include <library/unittest/registar.h>

#include <list>

namespace NThreading {

////////////////////////////////////////////////////////////////////////////////

SIMPLE_UNIT_TEST_SUITE(TFutureTest) {
    SIMPLE_UNIT_TEST(ShouldInitiallyHasNoValue)
    {
        TPromise<int> promise;
        UNIT_ASSERT(!promise.HasValue());

        promise = NewPromise<int>();
        UNIT_ASSERT(!promise.HasValue());

        TFuture<int> future;
        UNIT_ASSERT(!future.HasValue());

        future = promise.GetFuture();
        UNIT_ASSERT(!future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldInitiallyHasNoValueVoid)
    {
        TPromise<void> promise;
        UNIT_ASSERT(!promise.HasValue());

        promise = NewPromise();
        UNIT_ASSERT(!promise.HasValue());

        TFuture<void> future;
        UNIT_ASSERT(!future.HasValue());

        future = promise.GetFuture();
        UNIT_ASSERT(!future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldStoreValue)
    {
        TPromise<int> promise = NewPromise<int>();
        promise.SetValue(123);
        UNIT_ASSERT(promise.HasValue());
        UNIT_ASSERT_EQUAL(promise.GetValue(), 123);

        TFuture<int> future = promise.GetFuture();
        UNIT_ASSERT(future.HasValue());
        UNIT_ASSERT_EQUAL(future.GetValue(), 123);

        future = MakeFuture(345);
        UNIT_ASSERT(future.HasValue());
        UNIT_ASSERT_EQUAL(future.GetValue(), 345);
    }

    SIMPLE_UNIT_TEST(ShouldStoreValueVoid)
    {
        TPromise<void> promise = NewPromise();
        promise.SetValue();
        UNIT_ASSERT(promise.HasValue());

        TFuture<void> future = promise.GetFuture();
        UNIT_ASSERT(future.HasValue());

        future = MakeFuture();
        UNIT_ASSERT(future.HasValue());
    }

    struct TTestCallback {
        int Value;

        TTestCallback(int value)
            : Value(value)
        {}

        void Callback(const TFuture<int>& future)
        {
            Value += future.GetValue();
        }

        int Func(const TFuture<int>& future)
        {
            return (Value += future.GetValue());
        }

        void VoidFunc(const TFuture<int>& future)
        {
            future.GetValue();
        }

        TFuture<int> FutureFunc(const TFuture<int>& future)
        {
            return MakeFuture(Value += future.GetValue());
        }

        TPromise<void> Signal = NewPromise();
        TFuture<void> FutureVoidFunc(const TFuture<int>& future)
        {
            future.GetValue();
            return Signal;
        }
    };

    SIMPLE_UNIT_TEST(ShouldInvokeCallback)
    {
        TPromise<int> promise = NewPromise<int>();

        TTestCallback callback(123);
        TFuture<int> future = promise.GetFuture()
            .Subscribe([&](const TFuture<int>& theFuture){ return callback.Callback(theFuture); });

        promise.SetValue(456);
        UNIT_ASSERT_EQUAL(future.GetValue(), 456);
        UNIT_ASSERT_EQUAL(callback.Value, 123+456);
    }

    SIMPLE_UNIT_TEST(ShouldApplyFunc)
    {
        TPromise<int> promise = NewPromise<int>();

        TTestCallback callback(123);
        TFuture<int> future = promise.GetFuture()
            .Apply([&](const TFuture<int>& theFuture){ return callback.Func(theFuture); });

        promise.SetValue(456);
        UNIT_ASSERT_EQUAL(future.GetValue(), 123+456);
        UNIT_ASSERT_EQUAL(callback.Value, 123+456);
    }

    SIMPLE_UNIT_TEST(ShouldApplyVoidFunc)
    {
        TPromise<int> promise = NewPromise<int>();

        TTestCallback callback(123);
        TFuture<void> future = promise.GetFuture()
            .Apply([&](const TFuture<int>& theFuture){ return callback.VoidFunc(theFuture); });

        promise.SetValue(456);
        UNIT_ASSERT(future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldApplyFutureFunc)
    {
        TPromise<int> promise = NewPromise<int>();

        TTestCallback callback(123);
        TFuture<int> future = promise.GetFuture()
            .Apply([&](const TFuture<int>& theFuture){ return callback.FutureFunc(theFuture); });

        promise.SetValue(456);
        UNIT_ASSERT_EQUAL(future.GetValue(), 123+456);
        UNIT_ASSERT_EQUAL(callback.Value, 123+456);
    }

    SIMPLE_UNIT_TEST(ShouldApplyFutureVoidFunc)
    {
        TPromise<int> promise = NewPromise<int>();

        TTestCallback callback(123);
        TFuture<void> future = promise.GetFuture()
            .Apply([&](const TFuture<int>& theFuture){ return callback.FutureVoidFunc(theFuture); });

        promise.SetValue(456);
        UNIT_ASSERT(!future.HasValue());

        callback.Signal.SetValue();
        UNIT_ASSERT(future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldIgnoreResultIfAsked)
    {
        TPromise<int> promise = NewPromise<int>();

        TTestCallback callback(123);
        TFuture<int> future = promise.GetFuture().IgnoreResult().Return(42);

        promise.SetValue(456);
        UNIT_ASSERT_EQUAL(future.GetValue(), 42);
    }

    class TCustomException : public yexception {
    };

    SIMPLE_UNIT_TEST(ShouldRethrowException)
    {
        TPromise<int> promise = NewPromise<int>();
        try {
            ythrow TCustomException();
        } catch (...) {
            promise.SetException(std::current_exception());
        }

        UNIT_ASSERT(!promise.HasValue());
        UNIT_ASSERT(promise.HasException());
        UNIT_ASSERT_EXCEPTION(promise.GetValue(), TCustomException);
    }

    SIMPLE_UNIT_TEST(ShouldWaitAll)
    {
        TPromise<void> promise1 = NewPromise();
        TPromise<void> promise2 = NewPromise();

        TFuture<void> future = WaitAll(promise1, promise2);
        UNIT_ASSERT(!future.HasValue());

        promise1.SetValue();
        UNIT_ASSERT(!future.HasValue());

        promise2.SetValue();
        UNIT_ASSERT(future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldWaitAllVector)
    {
        TPromise<void> promise1 = NewPromise();
        TPromise<void> promise2 = NewPromise();

        yvector<TFuture<void>> promises;
        promises.push_back(promise1);
        promises.push_back(promise2);

        TFuture<void> future = WaitAll(promises);
        UNIT_ASSERT(!future.HasValue());

        promise1.SetValue();
        UNIT_ASSERT(!future.HasValue());

        promise2.SetValue();
        UNIT_ASSERT(future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldWaitAllList)
    {
        TPromise<void> promise1 = NewPromise();
        TPromise<void> promise2 = NewPromise();

        std::list<TFuture<void>> promises;
        promises.push_back(promise1);
        promises.push_back(promise2);

        TFuture<void> future = WaitAll(promises);
        UNIT_ASSERT(!future.HasValue());

        promise1.SetValue();
        UNIT_ASSERT(!future.HasValue());

        promise2.SetValue();
        UNIT_ASSERT(future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldWaitAllVectorEmpty)
    {
        yvector<TFuture<void>> promises;

        TFuture<void> future = WaitAll(promises);
        UNIT_ASSERT(future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldWaitAnyVector)
    {
        TPromise<void> promise1 = NewPromise();
        TPromise<void> promise2 = NewPromise();

        yvector<TFuture<void>> promises;
        promises.push_back(promise1);
        promises.push_back(promise2);

        TFuture<void> future = WaitAny(promises);
        UNIT_ASSERT(!future.HasValue());

        promise1.SetValue();
        UNIT_ASSERT(future.HasValue());

        promise2.SetValue();
        UNIT_ASSERT(future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldWaitAnyList)
    {
        TPromise<void> promise1 = NewPromise();
        TPromise<void> promise2 = NewPromise();

        std::list<TFuture<void>> promises;
        promises.push_back(promise1);
        promises.push_back(promise2);

        TFuture<void> future = WaitAny(promises);
        UNIT_ASSERT(!future.HasValue());

        promise1.SetValue();
        UNIT_ASSERT(future.HasValue());

        promise2.SetValue();
        UNIT_ASSERT(future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldWaitAnyVectorEmpty)
    {
        yvector<TFuture<void>> promises;

        TFuture<void> future = WaitAny(promises);
        UNIT_ASSERT(future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldWaitAny)
    {
        TPromise<void> promise1 = NewPromise();
        TPromise<void> promise2 = NewPromise();

        TFuture<void> future = WaitAny(promise1, promise2);
        UNIT_ASSERT(!future.HasValue());

        promise1.SetValue();
        UNIT_ASSERT(future.HasValue());

        promise2.SetValue();
        UNIT_ASSERT(future.HasValue());
    }

    SIMPLE_UNIT_TEST(ShouldStoreTypesWithoutDefaultConstructor)
    {
        // compileability test
        struct TRec {
            explicit TRec(int) {}
        };

        auto promise = NewPromise<TRec>();
        promise.SetValue(TRec(1));

        auto future = MakeFuture(TRec(1));
        const auto& rec = future.GetValue();
        Y_UNUSED(rec);
    }

    SIMPLE_UNIT_TEST(ShouldStoreMovableTypes)
    {
        // compileability test
        struct TRec: TMoveOnly {
            explicit TRec(int) {}
        };

        auto promise = NewPromise<TRec>();
        promise.SetValue(TRec(1));

        auto future = MakeFuture(TRec(1));
        const auto& rec = future.GetValue();
        Y_UNUSED(rec);
    }
}

}   // namespace NThreading
