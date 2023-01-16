#include "stat.h"

#include <util/generic/hash.h>
#include <util/generic/singleton.h>
#include <util/system/spinlock.h>
#include <util/system/tls.h>

using namespace NNeh;

volatile TAtomic NNeh::TServiceStat::MaxContinuousErrors_ = 0; //by default disabled
volatile TAtomic NNeh::TServiceStat::ReSendValidatorPeriod_ = 100;

NNeh::TServiceStat::EStatus NNeh::TServiceStat::GetStatus() {
    if (!AtomicGet(MaxContinuousErrors_) || AtomicGet(LastContinuousErrors_) < AtomicGet(MaxContinuousErrors_)) {
        return Ok;
    }

    if (RequestsInProcess_.Val() != 0)
        return Fail;

    if (AtomicIncrement(SendValidatorCounter_) != AtomicGet(ReSendValidatorPeriod_)) {
        return Fail;
    }

    //time for refresh service status (send validation request)
    AtomicSet(SendValidatorCounter_, 0);

    return ReTry;
}

void NNeh::TServiceStat::DbgOut(IOutputStream& out) const {
    out << "----------------------------------------------------" << '\n';;
    out << "RequestsInProcess: " << RequestsInProcess_.Val() << '\n';
    out << "LastContinuousErrors: " << AtomicGet(LastContinuousErrors_) << '\n';
    out << "SendValidatorCounter: " << AtomicGet(SendValidatorCounter_) << '\n';
    out << "ReSendValidatorPeriod: " << AtomicGet(ReSendValidatorPeriod_) << Endl;
}

void NNeh::TServiceStat::OnBegin() {
    RequestsInProcess_.Inc();
}

void NNeh::TServiceStat::OnSuccess() {
    RequestsInProcess_.Dec();
    AtomicSet(LastContinuousErrors_, 0);
}

void NNeh::TServiceStat::OnCancel() {
    RequestsInProcess_.Dec();
}

void NNeh::TServiceStat::OnFail() {
    RequestsInProcess_.Dec();
    if (AtomicIncrement(LastContinuousErrors_) == AtomicGet(MaxContinuousErrors_)) {
        AtomicSet(SendValidatorCounter_, 0);
    }
}

namespace {
    class TGlobalServicesStat {
    public:
        inline TServiceStatRef ServiceStat(const TStringBuf addr) noexcept {
            const auto guard = Guard(Lock_);

            TServiceStatRef& ss = SS_[addr];

            if (!ss) {
                TServiceStatRef tmp(new TServiceStat());

                ss.Swap(tmp);
            }
            return ss;
        }

    protected:
        TAdaptiveLock Lock_;
        THashMap<TString, TServiceStatRef> SS_;
    };

    class TServicesStat {
    public:
        inline TServiceStatRef ServiceStat(const TStringBuf addr) noexcept {
            TServiceStatRef& ss = SS_[addr];

            if (!ss) {
                TServiceStatRef tmp(Singleton<TGlobalServicesStat>()->ServiceStat(addr));

                ss.Swap(tmp);
            }
            return ss;
        }

    protected:
        THashMap<TString, TServiceStatRef> SS_;
    };

    inline TServicesStat* ThrServiceStat() {
        Y_POD_STATIC_THREAD(TServicesStat*)
        ss;

        if (!ss) {
            Y_STATIC_THREAD(TServicesStat)
            tss;

            ss = &(TServicesStat&)tss;
        }

        return ss;
    }
}

TServiceStatRef NNeh::GetServiceStat(const TStringBuf addr) {
    return ThrServiceStat()->ServiceStat(addr);
}
