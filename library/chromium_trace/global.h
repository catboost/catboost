#pragma once

#include "tracer.h"
#include "sampler.h"

#include <util/stream/file.h>

namespace NChromiumTrace {
    TTracer* GetGlobalTracer();
    TSamplerThread* GetGlobalSampler();

    class TGlobalTraceConsumerGuard {
        THolder<ITraceConsumer> Consumer;

    public:
        TGlobalTraceConsumerGuard(THolder<ITraceConsumer> consumer);
        ~TGlobalTraceConsumerGuard();
    };

    class TGlobalJsonFileSink {
        TFileOutput File;
        TGlobalTraceConsumerGuard Guard;

    public:
        TGlobalJsonFileSink(const TString& filename);
        ~TGlobalJsonFileSink();
    };

}
