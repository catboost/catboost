#include "async.h"

#include <library/unittest/registar.h>

#include <util/generic/ptr.h>
#include <util/generic/vector.h>

namespace {

struct TMySuperTaskQueue {
};

}

namespace NThreading {

/* Here we provide an Async overload for TMySuperTaskQueue indide NThreading namespace
 * so that we can call it in the way
 *
 * TMySuperTaskQueue queue;
 * NThreading::Async([](){}, queue);
 *
 * See also ExtensionExample unittest.
 */
template <typename Func>
TFuture<TFunctionResult<Func>> Async(Func func, TMySuperTaskQueue&) {
    return MakeFuture(func());
}

}

SIMPLE_UNIT_TEST_SUITE(Async) {

SIMPLE_UNIT_TEST(ExtensionExample) {
    TMySuperTaskQueue queue;
    auto future = NThreading::Async([]() { return 5; }, queue);
    future.Wait();
    UNIT_ASSERT_VALUES_EQUAL(future.GetValue(), 5);
}

SIMPLE_UNIT_TEST(WorksWithIMtpQueue) {
    auto queue = MakeHolder<TMtpQueue>();
    queue->Start(1);

    auto future = NThreading::Async([]() { return 5; }, *queue);
    future.Wait();
    UNIT_ASSERT_VALUES_EQUAL(future.GetValue(), 5);
}

SIMPLE_UNIT_TEST(ProperlyDeducesFutureType) {
    // Compileability test
    auto queue = CreateMtpQueue(1);

    NThreading::TFuture<void> f1 = NThreading::Async([](){}, *queue);
    NThreading::TFuture<int> f2 = NThreading::Async([](){ return 5; }, *queue);
    NThreading::TFuture<double> f3 = NThreading::Async([](){ return 5.0; }, *queue);
    NThreading::TFuture<yvector<int>> f4 = NThreading::Async([](){ return yvector<int>(); }, *queue);
    NThreading::TFuture<int> f5 = NThreading::Async([](){ return NThreading::MakeFuture(5); }, *queue);
}

}
