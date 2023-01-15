#pragma once

#include <util/generic/vector.h>
#include <util/system/condvar.h>
#include <util/system/thread.h>

#include <functional>

namespace NChromiumTrace {
    class TTracer;

    using TSamplerFunction = std::function<void(TTracer&)>;

    class TSamplerThread: public ISimpleThread {
        bool Continue;
        TMutex SignalLock;
        TMutex SamplersLock;
        TCondVar CV;
        TDuration Interval;
        TTracer* Tracer;
        TVector<TSamplerFunction> Samplers;

    public:
        TSamplerThread(TTracer* tracer, TDuration interval);
        ~TSamplerThread() override;

        void AddSampler(TSamplerFunction sampler);

        void SetInterval(TDuration interval);

        using ISimpleThread::Start;
        void Stop();

    private:
        void* ThreadProc() noexcept override;
        void RunSamplers();
    };

    class TSamplerHolder: public THolder<TSamplerThread> {
    public:
        TSamplerHolder(TTracer* tracer, TDuration interval);
        ~TSamplerHolder();
    };
}
