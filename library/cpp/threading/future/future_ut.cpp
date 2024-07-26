#include "future.h"

#include <library/cpp/testing/unittest/registar.h>

#include <list>
#include <type_traits>

namespace NThreading {

namespace {

    class TCopyCounter {
    public:
        TCopyCounter(size_t* numCopies)
            : NumCopies(numCopies)
        {}

        TCopyCounter(const TCopyCounter& that)
            : NumCopies(that.NumCopies)
        {
            ++*NumCopies;
        }

        TCopyCounter& operator=(const TCopyCounter& that) {
            NumCopies = that.NumCopies;
            ++*NumCopies;
            return *this;
        }

        TCopyCounter(TCopyCounter&& that) = default;

        TCopyCounter& operator=(TCopyCounter&& that) = default;

    private:
        size_t* NumCopies = nullptr;
    };

    template <typename T>
    auto MakePromise() {
        if constexpr (std::is_same_v<T, void>) {
            return NewPromise();
        }
        return NewPromise<T>();
    }


    template <typename T>
    void TestFutureStateId() {
        TFuture<T> empty;
        UNIT_ASSERT(!empty.StateId().Defined());
        auto promise1 = MakePromise<T>();
        auto future11 = promise1.GetFuture();
        UNIT_ASSERT(future11.StateId().Defined());
        auto future12 = promise1.GetFuture();
        UNIT_ASSERT_EQUAL(future11.StateId(), future11.StateId()); // same result for subsequent invocations
        UNIT_ASSERT_EQUAL(future11.StateId(), future12.StateId()); // same result for different futures with the same state
        auto promise2 = MakePromise<T>();
        auto future2 = promise2.GetFuture();
        UNIT_ASSERT(future2.StateId().Defined());
        UNIT_ASSERT_UNEQUAL(future11.StateId(), future2.StateId()); // different results for futures with different states
    }

}

    ////////////////////////////////////////////////////////////////////////////////

    Y_UNIT_TEST_SUITE(TFutureTest) {
        Y_UNIT_TEST(ShouldInitiallyHasNoValue) {
            TPromise<int> promise;
            UNIT_ASSERT(!promise.HasValue());

            promise = NewPromise<int>();
            UNIT_ASSERT(!promise.HasValue());

            TFuture<int> future;
            UNIT_ASSERT(!future.HasValue());

            future = promise.GetFuture();
            UNIT_ASSERT(!future.HasValue());
        }

        Y_UNIT_TEST(ShouldInitiallyHasNoValueVoid) {
            TPromise<void> promise;
            UNIT_ASSERT(!promise.HasValue());

            promise = NewPromise();
            UNIT_ASSERT(!promise.HasValue());

            TFuture<void> future;
            UNIT_ASSERT(!future.HasValue());

            future = promise.GetFuture();
            UNIT_ASSERT(!future.HasValue());
        }

        Y_UNIT_TEST(ShouldStoreValue) {
            TPromise<int> promise = NewPromise<int>();
            promise.SetValue(123);
            UNIT_ASSERT(promise.HasValue());
            UNIT_ASSERT_EQUAL(promise.GetValue(), 123);

            TFuture<int> future = promise.GetFuture();
            UNIT_ASSERT(future.HasValue());
            UNIT_ASSERT_EQUAL(future.GetValue(), 123);

            future = MakeFuture(345);
            UNIT_ASSERT(future.HasValue());
            UNIT_ASSERT(future.IsReady());
            UNIT_ASSERT_EQUAL(future.GetValue(), 345);
        }

        Y_UNIT_TEST(ShouldStoreValueVoid) {
            TPromise<void> promise = NewPromise();
            promise.SetValue();
            UNIT_ASSERT(promise.HasValue());

            TFuture<void> future = promise.GetFuture();
            UNIT_ASSERT(future.HasValue());
            UNIT_ASSERT(future.IsReady());

            future = MakeFuture();
            UNIT_ASSERT(future.HasValue());
        }

        struct TTestCallback {
            int Value;

            TTestCallback(int value)
                : Value(value)
            {
            }

            void Callback(const TFuture<int>& future) {
                Value += future.GetValue();
            }

            int Func(const TFuture<int>& future) {
                return (Value += future.GetValue());
            }

            void VoidFunc(const TFuture<int>& future) {
                future.GetValue();
            }

            TFuture<int> FutureFunc(const TFuture<int>& future) {
                return MakeFuture(Value += future.GetValue());
            }

            TPromise<void> Signal = NewPromise();
            TFuture<void> FutureVoidFunc(const TFuture<int>& future) {
                future.GetValue();
                return Signal;
            }
        };

        Y_UNIT_TEST(ShouldInvokeCallback) {
            TPromise<int> promise = NewPromise<int>();

            TTestCallback callback(123);
            TFuture<int> future = promise.GetFuture()
                                      .Subscribe([&](const TFuture<int>& theFuture) { return callback.Callback(theFuture); });

            promise.SetValue(456);
            UNIT_ASSERT_EQUAL(future.GetValue(), 456);
            UNIT_ASSERT_EQUAL(callback.Value, 123 + 456);
        }

        Y_UNIT_TEST(ShouldApplyFunc) {
            TPromise<int> promise = NewPromise<int>();

            TTestCallback callback(123);
            TFuture<int> future = promise.GetFuture()
                                      .Apply([&](const auto& theFuture) { return callback.Func(theFuture); });

            promise.SetValue(456);
            UNIT_ASSERT_EQUAL(future.GetValue(), 123 + 456);
            UNIT_ASSERT_EQUAL(callback.Value, 123 + 456);
        }

        Y_UNIT_TEST(ShouldApplyVoidFunc) {
            TPromise<int> promise = NewPromise<int>();

            TTestCallback callback(123);
            TFuture<void> future = promise.GetFuture()
                                       .Apply([&](const auto& theFuture) { return callback.VoidFunc(theFuture); });

            promise.SetValue(456);
            UNIT_ASSERT(future.HasValue());
        }

        Y_UNIT_TEST(ShouldApplyFutureFunc) {
            TPromise<int> promise = NewPromise<int>();

            TTestCallback callback(123);
            TFuture<int> future = promise.GetFuture()
                                      .Apply([&](const auto& theFuture) { return callback.FutureFunc(theFuture); });

            promise.SetValue(456);
            UNIT_ASSERT_EQUAL(future.GetValue(), 123 + 456);
            UNIT_ASSERT_EQUAL(callback.Value, 123 + 456);
        }

        Y_UNIT_TEST(ShouldApplyFutureVoidFunc) {
            TPromise<int> promise = NewPromise<int>();

            TTestCallback callback(123);
            TFuture<void> future = promise.GetFuture()
                                       .Apply([&](const auto& theFuture) { return callback.FutureVoidFunc(theFuture); });

            promise.SetValue(456);
            UNIT_ASSERT(!future.HasValue());

            callback.Signal.SetValue();
            UNIT_ASSERT(future.HasValue());
        }

        Y_UNIT_TEST(ShouldIgnoreResultIfAsked) {
            TPromise<int> promise = NewPromise<int>();

            TTestCallback callback(123);
            TFuture<int> future = promise.GetFuture().IgnoreResult().Return(42);

            promise.SetValue(456);
            UNIT_ASSERT_EQUAL(future.GetValue(), 42);
        }

        class TCustomException: public yexception {
        };

        Y_UNIT_TEST(ShouldRethrowException) {
            TPromise<int> promise = NewPromise<int>();
            try {
                ythrow TCustomException();
            } catch (...) {
                promise.SetException(std::current_exception());
            }

            UNIT_ASSERT(!promise.HasValue());
            UNIT_ASSERT(promise.HasException());
            UNIT_ASSERT_EXCEPTION(promise.GetValue(), TCustomException);
            UNIT_ASSERT_EXCEPTION(promise.TryRethrow(), TCustomException);
        }

        Y_UNIT_TEST(ShouldRethrowCallbackException) {
            TPromise<int> promise = NewPromise<int>();
            TFuture<int> future = promise.GetFuture();
            future.Subscribe([](const TFuture<int>&) {
                throw TCustomException();
            });

            UNIT_ASSERT_EXCEPTION(promise.SetValue(123), TCustomException);
        }

        Y_UNIT_TEST(ShouldRethrowCallbackExceptionIgnoreResult) {
            TPromise<int> promise = NewPromise<int>();
            TFuture<void> future = promise.GetFuture().IgnoreResult();
            future.Subscribe([](const TFuture<void>&) {
                throw TCustomException();
            });

            UNIT_ASSERT_EXCEPTION(promise.SetValue(123), TCustomException);
        }


        Y_UNIT_TEST(ShouldWaitExceptionOrAll) {
            TPromise<void> promise1 = NewPromise();
            TPromise<void> promise2 = NewPromise();

            TFuture<void> future = WaitExceptionOrAll(promise1, promise2);
            UNIT_ASSERT(!future.HasValue());

            promise1.SetValue();
            UNIT_ASSERT(!future.HasValue());

            promise2.SetValue();
            UNIT_ASSERT(future.HasValue());
        }

        Y_UNIT_TEST(ShouldWaitExceptionOrAllVector) {
            TPromise<void> promise1 = NewPromise();
            TPromise<void> promise2 = NewPromise();

            TVector<TFuture<void>> promises;
            promises.push_back(promise1);
            promises.push_back(promise2);

            TFuture<void> future = WaitExceptionOrAll(promises);
            UNIT_ASSERT(!future.HasValue());

            promise1.SetValue();
            UNIT_ASSERT(!future.HasValue());

            promise2.SetValue();
            UNIT_ASSERT(future.HasValue());
        }

        Y_UNIT_TEST(ShouldWaitExceptionOrAllVectorWithValueType) {
            TPromise<int> promise1 = NewPromise<int>();
            TPromise<int> promise2 = NewPromise<int>();

            TVector<TFuture<int>> promises;
            promises.push_back(promise1);
            promises.push_back(promise2);

            TFuture<void> future = WaitExceptionOrAll(promises);
            UNIT_ASSERT(!future.HasValue());

            promise1.SetValue(0);
            UNIT_ASSERT(!future.HasValue());

            promise2.SetValue(0);
            UNIT_ASSERT(future.HasValue());
        }

        Y_UNIT_TEST(ShouldWaitExceptionOrAllList) {
            TPromise<void> promise1 = NewPromise();
            TPromise<void> promise2 = NewPromise();

            std::list<TFuture<void>> promises;
            promises.push_back(promise1);
            promises.push_back(promise2);

            TFuture<void> future = WaitExceptionOrAll(promises);
            UNIT_ASSERT(!future.HasValue());

            promise1.SetValue();
            UNIT_ASSERT(!future.HasValue());

            promise2.SetValue();
            UNIT_ASSERT(future.HasValue());
        }

        Y_UNIT_TEST(ShouldWaitExceptionOrAllVectorEmpty) {
            TVector<TFuture<void>> promises;

            TFuture<void> future = WaitExceptionOrAll(promises);
            UNIT_ASSERT(future.HasValue());
        }

        Y_UNIT_TEST(ShouldWaitAnyVector) {
            TPromise<void> promise1 = NewPromise();
            TPromise<void> promise2 = NewPromise();

            TVector<TFuture<void>> promises;
            promises.push_back(promise1);
            promises.push_back(promise2);

            TFuture<void> future = WaitAny(promises);
            UNIT_ASSERT(!future.HasValue());

            promise1.SetValue();
            UNIT_ASSERT(future.HasValue());

            promise2.SetValue();
            UNIT_ASSERT(future.HasValue());
        }


        Y_UNIT_TEST(ShouldWaitAnyVectorWithValueType) {
            TPromise<int> promise1 = NewPromise<int>();
            TPromise<int> promise2 = NewPromise<int>();

            TVector<TFuture<int>> promises;
            promises.push_back(promise1);
            promises.push_back(promise2);

            TFuture<void> future = WaitAny(promises);
            UNIT_ASSERT(!future.HasValue());

            promise1.SetValue(0);
            UNIT_ASSERT(future.HasValue());

            promise2.SetValue(0);
            UNIT_ASSERT(future.HasValue());
        }

        Y_UNIT_TEST(ShouldWaitAnyList) {
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

        Y_UNIT_TEST(ShouldWaitAnyVectorEmpty) {
            TVector<TFuture<void>> promises;

            TFuture<void> future = WaitAny(promises);
            UNIT_ASSERT(future.HasValue());
        }

        Y_UNIT_TEST(ShouldWaitAny) {
            TPromise<void> promise1 = NewPromise();
            TPromise<void> promise2 = NewPromise();

            TFuture<void> future = WaitAny(promise1, promise2);
            UNIT_ASSERT(!future.HasValue());

            promise1.SetValue();
            UNIT_ASSERT(future.HasValue());

            promise2.SetValue();
            UNIT_ASSERT(future.HasValue());
        }

        Y_UNIT_TEST(ShouldStoreTypesWithoutDefaultConstructor) {
            // compileability test
            struct TRec {
                explicit TRec(int) {
                }
            };

            auto promise = NewPromise<TRec>();
            promise.SetValue(TRec(1));

            auto future = MakeFuture(TRec(1));
            const auto& rec = future.GetValue();
            Y_UNUSED(rec);
        }

        Y_UNIT_TEST(ShouldStoreMovableTypes) {
            // compileability test
            struct TRec : TMoveOnly {
                explicit TRec(int) {
                }
            };

            auto promise = NewPromise<TRec>();
            promise.SetValue(TRec(1));

            auto future = MakeFuture(TRec(1));
            const auto& rec = future.GetValue();
            Y_UNUSED(rec);
        }

        Y_UNIT_TEST(ShouldMoveMovableTypes) {
            // compileability test
            struct TRec : TMoveOnly {
                explicit TRec(int) {
                }
            };

            auto promise = NewPromise<TRec>();
            promise.SetValue(TRec(1));

            auto future = MakeFuture(TRec(1));
            auto rec = future.ExtractValue();
            Y_UNUSED(rec);
        }

        Y_UNIT_TEST(ShouldNotExtractAfterGet) {
            TPromise<int> promise = NewPromise<int>();
            promise.SetValue(123);
            UNIT_ASSERT(promise.HasValue());
            UNIT_ASSERT_EQUAL(promise.GetValue(), 123);
            UNIT_CHECK_GENERATED_EXCEPTION(promise.ExtractValue(), TFutureException);
        }

        Y_UNIT_TEST(ShouldNotGetAfterExtract) {
            TPromise<int> promise = NewPromise<int>();
            promise.SetValue(123);
            UNIT_ASSERT(promise.HasValue());
            UNIT_ASSERT_EQUAL(promise.ExtractValue(), 123);
            UNIT_CHECK_GENERATED_EXCEPTION(promise.GetValue(), TFutureException);
        }

        Y_UNIT_TEST(ShouldNotExtractAfterExtract) {
            TPromise<int> promise = NewPromise<int>();
            promise.SetValue(123);
            UNIT_ASSERT(promise.HasValue());
            UNIT_ASSERT_EQUAL(promise.ExtractValue(), 123);
            UNIT_CHECK_GENERATED_EXCEPTION(promise.ExtractValue(), TFutureException);
        }

        Y_UNIT_TEST(ShouldNotExtractFromSharedDefault) {
            UNIT_CHECK_GENERATED_EXCEPTION(MakeFuture<int>().ExtractValue(), TFutureException);

            struct TStorage {
                TString String = TString(100, 'a');
            };
            try {
                TString s = MakeFuture<TStorage>().ExtractValue().String;
                Y_UNUSED(s);
            } catch (TFutureException) {
                // pass
            }
            UNIT_ASSERT_VALUES_EQUAL(MakeFuture<TStorage>().GetValue().String, TString(100, 'a'));
        }

        Y_UNIT_TEST(HandlingRepetitiveSet) {
            TPromise<int> promise = NewPromise<int>();
            promise.SetValue(42);
            UNIT_CHECK_GENERATED_EXCEPTION(promise.SetValue(42), TFutureException);
        }

        Y_UNIT_TEST(HandlingRepetitiveTrySet) {
            TPromise<int> promise = NewPromise<int>();
            UNIT_ASSERT(promise.TrySetValue(42));
            UNIT_ASSERT(!promise.TrySetValue(42));
        }

        Y_UNIT_TEST(HandlingRepetitiveSetException) {
            TPromise<int> promise = NewPromise<int>();
            promise.SetException("test");
            UNIT_CHECK_GENERATED_EXCEPTION(promise.SetException("test"), TFutureException);
        }

        Y_UNIT_TEST(HandlingRepetitiveTrySetException) {
            TPromise<int> promise = NewPromise<int>();
            UNIT_ASSERT(promise.TrySetException(std::make_exception_ptr("test")));
            UNIT_ASSERT(!promise.TrySetException(std::make_exception_ptr("test")));
        }

        Y_UNIT_TEST(ShouldAllowToMakeFutureWithException)
        {
            auto future1 = MakeErrorFuture<void>(std::make_exception_ptr(TFutureException()));
            UNIT_ASSERT(future1.HasException());
            UNIT_ASSERT(future1.IsReady());
            UNIT_CHECK_GENERATED_EXCEPTION(future1.GetValue(), TFutureException);

            auto future2 = MakeErrorFuture<int>(std::make_exception_ptr(TFutureException()));
            UNIT_ASSERT(future2.HasException());
            UNIT_CHECK_GENERATED_EXCEPTION(future2.GetValue(), TFutureException);

            auto future3 = MakeFuture<std::exception_ptr>(std::make_exception_ptr(TFutureException()));
            UNIT_ASSERT(future3.HasValue());
            UNIT_CHECK_GENERATED_NO_EXCEPTION(future3.GetValue(), TFutureException);

            auto future4 = MakeFuture<std::unique_ptr<int>>(nullptr);
            UNIT_ASSERT(future4.HasValue());
            UNIT_CHECK_GENERATED_NO_EXCEPTION(future4.GetValue(), TFutureException);
        }

        Y_UNIT_TEST(WaitAllowsExtract) {
            auto future = MakeFuture<int>(42);
            TVector vec{future, future, future};
            WaitExceptionOrAll(vec).GetValue();
            WaitAny(vec).GetValue();

            UNIT_ASSERT_EQUAL(future.ExtractValue(), 42);
        }

        Y_UNIT_TEST(IgnoreAllowsExtract) {
            auto future = MakeFuture<int>(42);
            future.IgnoreResult().GetValue();

            UNIT_ASSERT_EQUAL(future.ExtractValue(), 42);
        }

        Y_UNIT_TEST(WaitExceptionOrAllException) {
            auto promise1 = NewPromise();
            auto promise2 = NewPromise();
            auto future1 = promise1.GetFuture();
            auto future2 = promise2.GetFuture();
            auto wait = WaitExceptionOrAll(future1, future2);
            promise2.SetException("foo-exception");
            wait.Wait();
            UNIT_ASSERT(future2.HasException());
            UNIT_ASSERT(!future1.IsReady());
            UNIT_ASSERT(!future1.HasValue() && !future1.HasException());
        }

        Y_UNIT_TEST(WaitAllException) {
            auto promise1 = NewPromise();
            auto promise2 = NewPromise();
            auto future1 = promise1.GetFuture();
            auto future2 = promise2.GetFuture();
            auto wait = WaitAll(future1, future2);
            promise2.SetException("foo-exception");
            UNIT_ASSERT(!wait.HasValue() && !wait.HasException());
            promise1.SetValue();
            UNIT_ASSERT_EXCEPTION_CONTAINS(wait.GetValueSync(), yexception, "foo-exception");
        }

        Y_UNIT_TEST(FutureStateId) {
            TestFutureStateId<void>();
            TestFutureStateId<int>();
        }

        template <typename T>
        void TestApplyNoRvalueCopyImpl() {
            size_t numCopies = 0;
            TCopyCounter copyCounter(&numCopies);

            auto promise = MakePromise<T>();

            const auto future = promise.GetFuture().Apply(
                [copyCounter = std::move(copyCounter)] (const auto&) {}
            );

            if constexpr (std::is_same_v<T, void>) {
                promise.SetValue();
            } else {
                promise.SetValue(T());
            }

            future.GetValueSync();

            UNIT_ASSERT_VALUES_EQUAL(numCopies, 0);
        }

        Y_UNIT_TEST(ApplyNoRvalueCopy) {
            TestApplyNoRvalueCopyImpl<void>();
            TestApplyNoRvalueCopyImpl<int>();
        }

        template <typename T>
        void TestApplyLvalueCopyImpl() {
            size_t numCopies = 0;
            TCopyCounter copyCounter(&numCopies);

            auto promise = MakePromise<T>();

            auto func = [copyCounter = std::move(copyCounter)] (const auto&) {};
            const auto future = promise.GetFuture().Apply(func);

            if constexpr (std::is_same_v<T, void>) {
                promise.SetValue();
            } else {
                promise.SetValue(T());
            }

            future.GetValueSync();

            UNIT_ASSERT_VALUES_EQUAL(numCopies, 1);
        }

        Y_UNIT_TEST(ApplyLvalueCopy) {
            TestApplyLvalueCopyImpl<void>();
            TestApplyLvalueCopyImpl<int>();
        }

        Y_UNIT_TEST(ReturnForwardingTypeDeduction) {
            const TString e = TString(80, 'a');
            TString l = TString(80, 'a');

            TFuture<TString> futureL = MakeFuture().Return(l);
            UNIT_ASSERT_VALUES_EQUAL(futureL.GetValue(), e);
            UNIT_ASSERT_VALUES_EQUAL(l, e);

            TFuture<TString> futureR = MakeFuture().Return(std::move(l));
            UNIT_ASSERT_VALUES_EQUAL(futureR.GetValue(), e);
        }

        Y_UNIT_TEST(ReturnForwardingCopiesCount) {
            size_t numCopies = 0;
            TCopyCounter copyCounter(&numCopies);

            auto returnedCounter = MakeFuture().Return(std::move(copyCounter)).ExtractValueSync();
            Y_DO_NOT_OPTIMIZE_AWAY(returnedCounter);

            UNIT_ASSERT_VALUES_EQUAL(numCopies, 0);
        }
    }

}
