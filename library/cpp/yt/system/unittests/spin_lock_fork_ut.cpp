#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/rw_spin_lock.h>
#include <library/cpp/yt/system/fork_aware_spin_lock.h>

#include <util/thread/pool.h>

#include <sys/wait.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TReaderWriterSpinLockTest, ForkFriendlyness)
{
    std::atomic<bool> stopped = {false};
    YT_DECLARE_SPIN_LOCK(TReaderWriterSpinLock, lock);

    auto readerTask = [&lock, &stopped] () {
        while (!stopped.load()) {
            ForkFriendlyReaderGuard(lock);
        }
    };

    auto tryReaderTask = [&lock, &stopped] () {
        while (!stopped.load()) {
            // NB(pavook): TryAcquire instead of Acquire to minimize checks.
            bool acquired = lock.TryAcquireReaderForkFriendly();
            if (acquired) {
                lock.ReleaseReader();
            }
        }
    };

    auto tryWriterTask = [&lock, &stopped] () {
        while (!stopped.load()) {
            Sleep(TDuration::MicroSeconds(1));
            bool acquired = lock.TryAcquireWriter();
            if (acquired) {
                lock.ReleaseWriter();
            }
        }
    };

    auto writerTask = [&lock, &stopped] () {
        while (!stopped.load()) {
            Sleep(TDuration::MicroSeconds(1));
            WriterGuard(lock);
        }
    };

    int readerCount = 20;
    int writerCount = 10;

    auto reader = CreateThreadPool(readerCount);
    auto writer = CreateThreadPool(writerCount);

    for (int i = 0; i < readerCount / 2; ++i) {
        reader->SafeAddFunc(readerTask);
        reader->SafeAddFunc(tryReaderTask);
    }
    for (int i = 0; i < writerCount / 2; ++i) {
        writer->SafeAddFunc(writerTask);
        writer->SafeAddFunc(tryWriterTask);
    }

    // And let the chaos begin!
    int forkCount = 2000;
    for (int iter = 1; iter <= forkCount; ++iter) {
        pid_t pid;
        {
            auto guard = WriterGuard(lock);
            pid = fork();
        }

        YT_VERIFY(pid >= 0);

        // NB(pavook): check different orders to maximize chaos.
        if (iter % 2 == 0) {
            ReaderGuard(lock);
        }
        WriterGuard(lock);
        ReaderGuard(lock);
        if (pid == 0) {
            // NB(pavook): thread pools are no longer with us.
            _exit(0);
        }
    }

    for (int i = 1; i <= forkCount; ++i) {
        int status;
        YT_VERIFY(waitpid(0, &status, 0) > 0);
        YT_VERIFY(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }

    stopped.store(true);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TForkAwareSpinLockTest, ForkSafety)
{
    // TODO(pavook): reenable.
    GTEST_SKIP_("We are forced to use WriterStarvingRWSpinLock because of reentrancy problems - skipping the non-starvation test");

    std::atomic<bool> stopped = {false};
    YT_DECLARE_SPIN_LOCK(TForkAwareSpinLock, lock);

    auto acquireTask = [&lock, &stopped] () {
        while (!stopped.load()) {
            Guard(lock);
        }
    };

    // NB(pavook): TryAcquire instead of Acquire to minimize checks.
    auto tryAcquireTask = [&lock, &stopped] () {
        while (!stopped.load()) {
            bool acquired = lock.TryAcquire();
            if (acquired) {
                lock.Release();
            }
        }
    };

    int workerCount = 20;

    auto worker = CreateThreadPool(workerCount);

    for (int i = 0; i < workerCount / 2; ++i) {
        worker->SafeAddFunc(acquireTask);
        worker->SafeAddFunc(tryAcquireTask);
    }

    // And let the chaos begin!
    int forkCount = 2000;
    for (int iter = 1; iter <= forkCount; ++iter) {
        pid_t pid = fork();

        YT_VERIFY(pid >= 0);

        Guard(lock);
        Guard(lock);

        if (pid == 0) {
            // NB(pavook): thread pools are no longer with us.
            _exit(0);
        }
    }

    for (int i = 1; i <= forkCount; ++i) {
        int status;
        YT_VERIFY(waitpid(0, &status, 0) > 0);
        YT_VERIFY(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }

    stopped.store(true);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
