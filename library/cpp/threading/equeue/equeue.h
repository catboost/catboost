#pragma once

#include <util/thread/pool.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/generic/ptr.h>

//actual queue limit will be (maxQueueSize - numBusyThreads) or 0
class TElasticQueue: public IThreadPool {
public:
    explicit TElasticQueue(THolder<IThreadPool> slaveQueue);

    bool Add(IObjectInQueue* obj) override;
    size_t Size() const noexcept override;

    void Start(size_t threadCount, size_t maxQueueSize) override;
    void Stop() noexcept override;

    size_t ObjectCount() const;
private:
    class TDecrementingWrapper;

    bool TryIncCounter();
private:
    THolder<IThreadPool> SlaveQueue_;
    size_t MaxQueueSize_ = 0;
    TAtomic ObjectCount_ = 0;
    TAtomic GuardCount_ = 0;
};
