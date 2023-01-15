#include "wait_ut_common.h"

#include <util/random/shuffle.h>
#include <util/system/event.h>
#include <util/thread/pool.h>

namespace NThreading::NTest::NPrivate {

void ExecuteAndWait(TVector<std::function<void()>> jobs, TFuture<void> waiter, size_t threads) {
    Y_ENSURE(threads > 0);
    Shuffle(jobs.begin(), jobs.end());
    auto pool = CreateThreadPool(threads);
    TManualEvent start;
    for (auto& j : jobs) {
        pool->SafeAddFunc(
                        [&start, job = std::move(j)]() {
                            start.WaitI();
                            job();
                        });
    }
    start.Signal();
    waiter.Wait();
    pool->Stop();
}

}
