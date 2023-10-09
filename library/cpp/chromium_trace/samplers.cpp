#include "samplers.h"

#include "counter.h"

#include <util/generic/xrange.h>
#include <util/stream/file.h>
#include <util/string/builder.h>
#include <util/string/split.h>
#include <util/system/mem_info.h>

#if defined(_linux_) || defined(_darwin_)
#include <sys/resource.h>
#include <sys/time.h>
#endif

using namespace NChromiumTrace;

ISampler::~ISampler() = default;

void TSamplerBase::operator()(TTracer& tracer) {
    Update();
    Publish(tracer);
}

void TMemInfoSampler::Update() {
    auto memInfo = NMemInfo::GetMemInfo();
    RSS.Update(memInfo.RSS);
    VMS.Update(memInfo.VMS);
}

TProxySampler::TProxySampler(TIntrusivePtr<TSharedSamplerBase> impl)
    : Impl(impl)
{
    Y_ABORT_UNLESS(Impl);
}

void TProxySampler::Update() {
    Impl->Update();
}

void TProxySampler::Publish(TTracer& tracer) const {
    Impl->Publish(tracer);
}

void TMemInfoSampler::Publish(TTracer& tracer) const {
    TCounter("Memory.RSS")
        .Sample(TStringBuf("RSS"), RSS.Value)
        .Publish(tracer);
    TCounter("Memory.DeltaRSS")
        .Sample(TStringBuf("+"), RSS.PositiveDerivative)
        .Sample(TStringBuf("-"), RSS.NegativeDerivative)
        .Publish(tracer);
    TCounter("Memory.VMS")
        .Sample(TStringBuf("VMS"), VMS)
        .Publish(tracer);
}

#if defined(_linux_) || defined(_darwin_)
void TRUsageSampler::Update() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);

    UserTime.Update(TInstant(ru.ru_utime).MicroSeconds());
    SystemTime.Update(TInstant(ru.ru_stime).MicroSeconds());
    MinorFaults.Update(ru.ru_minflt);
    MajorFaults.Update(ru.ru_majflt);
    InputBlockIO.Update(ru.ru_inblock);
    OutputBlockIO.Update(ru.ru_oublock);
    VoluntarySwitches.Update(ru.ru_nvcsw);
    InvoluntarySwitches.Update(ru.ru_nivcsw);
}

void TRUsageSampler::Publish(TTracer& tracer) const {
    TCounter("RUsage.CPU")
        .Sample(TStringBuf("system"), SystemTime)
        .Sample(TStringBuf("user"), UserTime)
        .Publish(tracer);
    TCounter("RUsage.PageFaults")
        .Sample(TStringBuf("minor"), MinorFaults)
        .Sample(TStringBuf("major"), MajorFaults)
        .Publish(tracer);
    /*
    // FIXME: seems quite useless
    TCounter("RUsage.IO")
        .Sample("input", InputBlockIO)
        .Sample("output", OutputBlockIO)
        .Publish(tracer);

    */
    TCounter("RUsage.ContextSwitch")
        .Sample(TStringBuf("wait"), VoluntarySwitches)
        .Sample(TStringBuf("preempt"), InvoluntarySwitches)
        .Publish(tracer);
}

TNetStatSampler::TNetStatSampler(std::initializer_list<TString> keys)
    : AllowAllKeys(false)
{
    for (const auto& key : keys) {
        Values[key];
    }
}

void TNetStatSampler::Update() {
    // XXX: Actually, these stats are not per-process, but per-namespace
    UpdateFromData(TUnbufferedFileInput("/proc/self/net/snmp").ReadAll());
    UpdateFromData(TUnbufferedFileInput("/proc/self/net/netstat").ReadAll());
}

void TNetStatSampler::UpdateFromData(const TString& data) {
    TStringInput input(data);

    struct TNetStatEntry {
        TString Header;
        TVector<TString> Names;
        TVector<i64> Values;
    };

    TString line;
    while (input.ReadLine(line)) {
        TNetStatEntry entry;

        // Odd line: a list of names
        {
            auto pos = line.find(':');
            entry.Header = line.substr(0, pos);
            auto tail = TStringBuf(line.data() + pos + 2, line.data() + line.size());
            StringSplitter(tail).Split(' ').Consume([&](TStringBuf token) {
                entry.Names.emplace_back(token);
            });
        }

        // Even line : a list of values
        {
            input.ReadLine(line);
            auto pos = line.find(':');
            entry.Header = line.substr(0, pos);
            auto tail = TStringBuf(line.data() + pos + 2, line.data() + line.size());
            StringSplitter(tail).Split(' ').Consume([&](TStringBuf token) {
                entry.Values.emplace_back(FromString<i64>(token));
            });
        }

        for (size_t i : xrange(entry.Names.size())) {
            TString name = TStringBuilder() << entry.Header << '.' << entry.Names[i];
            auto it = Values.find(name);
            if (it != Values.end()) {
                it->second.Update(entry.Values[i]);
            } else if (AllowAllKeys) {
                Values[name].Update(entry.Values[i]);
            }
        }
    }
}

void TNetStatSampler::Publish(TTracer& tracer) const {
    for (const auto& item : Values) {
        TCounter(item.first, TStringBuf("sample"))
            .Sample(TStringBuf("value"), item.second)
            .Publish(tracer);
    }
}
#endif
