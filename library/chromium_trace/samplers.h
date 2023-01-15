#pragma once

#include "sample_value.h"

#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/system/guard.h>

namespace NChromiumTrace {
    class TTracer;

    class ISampler {
    public:
        virtual ~ISampler();

        virtual void Update() = 0;
        virtual void Publish(TTracer& tracer) const = 0;
    };

    /**
     * A base class providing \p operator() to allow conversion to
     * std::function<void(TTracer&)>.
     */
    class TSamplerBase: public ISampler {
    public:
        void operator()(TTracer& tracer);
    };

    /**
     * A sampler object is shared between source and sampler threads
     * and must provide thread-save Update(), Publish() methods.
     */
    class TSharedSamplerBase: public ISampler, public TThrRefBase {
    };

    /**
     * A sampler that shares ownership of another sampler and delegates
     * method calls to it.
     */
    class TProxySampler final: public TSamplerBase {
        TIntrusivePtr<TSharedSamplerBase> Impl;

    public:
        TProxySampler(TIntrusivePtr<TSharedSamplerBase> impl);

        void Update() override;
        void Publish(TTracer& tracer) const override;
    };

    class TMemInfoSampler final: public TSamplerBase {
        TDerivativeSampleValue<i64> RSS;
        TSampleValue<i64> VMS;

    public:
        void Update() override;
        void Publish(TTracer& tracer) const override;
    };

    class TRUsageSampler final: public TSamplerBase {
        TDerivativeSampleValue<i64> UserTime;
        TDerivativeSampleValue<i64> SystemTime;
        TDerivativeSampleValue<i64> MinorFaults;
        TDerivativeSampleValue<i64> MajorFaults;
        TDerivativeSampleValue<i64> InputBlockIO;
        TDerivativeSampleValue<i64> OutputBlockIO;
        TDerivativeSampleValue<i64> VoluntarySwitches;
        TDerivativeSampleValue<i64> InvoluntarySwitches;

    public:
        void Update() override;
        void Publish(TTracer& tracer) const override;
    };

    class TNetStatSampler final: public TSamplerBase {
        THashMap<TString, TDerivativeSampleValue<i64>> Values;
        bool AllowAllKeys = true;

    public:
        TNetStatSampler() = default;
        TNetStatSampler(std::initializer_list<TString> keys);

        void Update() override;
        void Publish(TTracer& tracer) const override;

    private:
        void UpdateFromData(const TString& data);
    };
}
