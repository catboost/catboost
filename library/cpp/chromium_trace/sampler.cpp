#include "sampler.h"

#include "guard.h"
#include "tracer.h"

using namespace NChromiumTrace;

TSamplerThread::TSamplerThread(TTracer* tracer, TDuration interval)
    : Continue(true)
    , Interval(interval)
    , Tracer(tracer)
{
    Y_ABORT_UNLESS(Tracer);
}

TSamplerThread::~TSamplerThread() = default;

void TSamplerThread::AddSampler(TSamplerFunction sampler) {
    with_lock (SamplersLock) {
        Samplers.push_back(sampler);
    }
}

void TSamplerThread::RunSamplers() {
    TCompleteEventGuard traceOverhead(
        Tracer,
        TStringBuf("RunSamplers"),
        TStringBuf("func,overhead"));

    for (const auto& sampler : Samplers) {
        try {
            sampler(*Tracer);
        } catch (...) {
        }
    }
}

void* TSamplerThread::ThreadProc() noexcept {
    Tracer->AddCurrentThreadName(TStringBuf("TraceSampler"));
    Tracer->AddCurrentThreadIndex(10000); // Stick it to the bottom

    while (true) {
        with_lock (SamplersLock) {
            RunSamplers();
        }
        with_lock (SignalLock) {
            if (!Continue) {
                break;
            }
            CV.WaitT(SignalLock, Interval);
        }
    }

    return nullptr;
}

void TSamplerThread::SetInterval(TDuration interval) {
    with_lock (SignalLock) {
        Interval = interval;
    }
}

void TSamplerThread::Stop() {
    with_lock (SignalLock) {
        Continue = false;
        CV.Signal();
    }
}

TSamplerHolder::TSamplerHolder(TTracer* tracer, TDuration interval)
    : THolder<TSamplerThread>(new TSamplerThread(tracer, interval))
{
}

TSamplerHolder::~TSamplerHolder() {
    if (auto* thread = Get()) {
        thread->Stop();
        thread->Join(); // FIXME: wtf? TSamplerThread should Join() in destructor
    }
}
