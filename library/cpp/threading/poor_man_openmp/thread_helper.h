#pragma once

#include <util/thread/pool.h>
#include <util/generic/utility.h>
#include <util/generic/yexception.h>
#include <util/system/info.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/condvar.h>
#include <util/system/mutex.h>
#include <util/stream/output.h>

#include <functional>
#include <cstdlib>

class TMtpQueueHelper {
public:
    TMtpQueueHelper() {
        SetThreadCount(NSystemInfo::CachedNumberOfCpus());
    }
    IThreadPool* Get() {
        return q.Get();
    }
    size_t GetThreadCount() {
        return ThreadCount;
    }
    void SetThreadCount(size_t threads) {
        ThreadCount = threads;
        q = CreateThreadPool(ThreadCount);
    }

    static TMtpQueueHelper& Instance();

private:
    size_t ThreadCount;
    TAutoPtr<IThreadPool> q;
};

namespace NYmp {
    inline void SetThreadCount(size_t threads) {
        TMtpQueueHelper::Instance().SetThreadCount(threads);
    }

    inline size_t GetThreadCount() {
        return TMtpQueueHelper::Instance().GetThreadCount();
    }

    template <typename T>
    inline void ParallelForStaticChunk(T begin, T end, size_t chunkSize, std::function<void(T)> func) {
        chunkSize = Max<size_t>(chunkSize, 1);

        size_t threadCount = TMtpQueueHelper::Instance().GetThreadCount();
        IThreadPool* queue = TMtpQueueHelper::Instance().Get();
        TCondVar cv;
        TMutex mutex;
        TAtomic counter = threadCount;
        std::exception_ptr err;

        for (size_t i = 0; i < threadCount; ++i) {
            queue->SafeAddFunc([&cv, &counter, &mutex, &func, i, begin, end, chunkSize, threadCount, &err]() {
                try {
                    T currentChunkStart = begin + static_cast<decltype(T() - T())>(i * chunkSize);

                    while (currentChunkStart < end) {
                        T currentChunkEnd = Min<T>(end, currentChunkStart + chunkSize);

                        for (T val = currentChunkStart; val < currentChunkEnd; ++val) {
                            func(val);
                        }

                        currentChunkStart += chunkSize * threadCount;
                    }
                } catch (...) {
                    with_lock (mutex) {
                        err = std::current_exception();
                    }
                }

                with_lock (mutex) {
                    if (AtomicDecrement(counter) == 0) {
                        //last one
                        cv.Signal();
                    }
                }
            });
        }

        with_lock (mutex) {
            while (AtomicGet(counter) > 0) {
                cv.WaitI(mutex);
            }
        }

        if (err) {
            std::rethrow_exception(err);
        }
    }

    template <typename T>
    inline void ParallelForStaticAutoChunk(T begin, T end, std::function<void(T)> func) {
        const size_t taskSize = end - begin;
        const size_t threadCount = TMtpQueueHelper::Instance().GetThreadCount();

        ParallelForStaticChunk(begin, end, (taskSize + threadCount - 1) / threadCount, func);
    }
}
