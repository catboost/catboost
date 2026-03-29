#pragma once

#include <util/thread/pool.h>
#include <util/generic/ptr.h>

#include <atomic>

//actual queue limit will be (maxQueueSize - numBusyThreads) or 0
class TElasticQueue: public IThreadPool {
public:
    explicit TElasticQueue(THolder<IThreadPool> slaveQueue);

    bool Add(IObjectInQueue* obj) override;
    size_t Size() const noexcept override;

    void Start(size_t threadCount, size_t maxQueueSize) override;
    void Stop() noexcept override;

    size_t ObjectCount() const;

    void SetCurrentMaxQueueSize(size_t v) {
        Y_ENSURE(v <= MaxQueueSize_);
        CurrentMaxQueueSize_ = v;
    }
private:
    class TDecrementingWrapper;

    bool TryIncCounter();
private:
    THolder<IThreadPool> SlaveQueue_;

    size_t MaxQueueSize_ = 0;
    std::atomic<size_t> CurrentMaxQueueSize_ = 0;
    std::atomic<size_t> ObjectCount_ = 0;
    std::atomic<size_t> GuardCount_ = 0;
};
