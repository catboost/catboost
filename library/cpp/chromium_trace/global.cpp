#include "global.h"

#include "json.h"
#include "sync.h"
#include "sampler.h"

#include <library/cpp/json/json_writer.h>

#include <util/generic/singleton.h>
#include <util/stream/file.h>

namespace {
    struct TGlobalSampler {
        NChromiumTrace::TSamplerHolder Sampler;

        TGlobalSampler()
            : Sampler(NChromiumTrace::GetGlobalTracer(), TDuration::Seconds(1))
        {
        }
    };
}

namespace NChromiumTrace {
    TTracer* GetGlobalTracer() {
        return Singleton<TTracer>();
    }

    TSamplerThread* GetGlobalSampler() {
        return Singleton<TGlobalSampler>()->Sampler.Get();
    }

    TGlobalTraceConsumerGuard::TGlobalTraceConsumerGuard(THolder<ITraceConsumer> consumer)
        : Consumer(std::move(consumer))
    {
        GetGlobalTracer()->SetOutput(Consumer.Get());
    }

    TGlobalTraceConsumerGuard::~TGlobalTraceConsumerGuard() {
        GetGlobalTracer()->SetOutput(nullptr);
    }

    TGlobalJsonFileSink::TGlobalJsonFileSink(const TString& filename)
        : File(filename)
        , Guard(MakeHolder<TSyncTraceConsumer<TJsonTraceConsumer>>(&File))
    {
    }

    TGlobalJsonFileSink::~TGlobalJsonFileSink() = default;

}
