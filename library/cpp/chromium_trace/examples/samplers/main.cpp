#include <library/cpp/chromium_trace/global.h>
#include <library/cpp/chromium_trace/json.h>
#include <library/cpp/chromium_trace/sync.h>
#include <library/cpp/chromium_trace/sampler.h>
#include <library/cpp/chromium_trace/samplers.h>

using namespace NChromiumTrace;

int main() {
    TGlobalTraceConsumerGuard guard(
        MakeHolder<TSyncTraceConsumer<TJsonTraceConsumer>>(&Cout));

    TSamplerHolder sampler(GetGlobalTracer(), TDuration::Seconds(1));
    sampler->AddSampler(TMemInfoSampler());
#if defined(_linux_) || defined(_darwin_)
    sampler->AddSampler(TRUsageSampler());
    sampler->AddSampler(TNetStatSampler());
#endif

    sampler->Start();
    while (true) {
        NanoSleep(1000000);
    }

    return 0;
}
