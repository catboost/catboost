#pragma once

#include <util/generic/ptr.h>
#include <util/stream/output.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <library/cpp/deprecated/atomic/atomic_ops.h>

namespace NNeh {
    class TStatCollector;

    /// NEH service workability statistics collector.
    ///
    /// Disabled by default, use `TServiceStat::ConfigureValidator` to set `maxContinuousErrors`
    /// different from zero.
    class TServiceStat: public TThrRefBase {
    public:
        static void ConfigureValidator(unsigned maxContinuousErrors, unsigned reSendValidatorPeriod) noexcept {
            AtomicSet(MaxContinuousErrors_, maxContinuousErrors);
            AtomicSet(ReSendValidatorPeriod_, reSendValidatorPeriod);
        }
        static bool Disabled() noexcept {
            return !AtomicGet(MaxContinuousErrors_);
        }

        enum EStatus {
            Ok,
            Fail,
            ReTry //time for sending request-validator to service
        };

        EStatus GetStatus();

        void DbgOut(IOutputStream&) const;

    protected:
        friend class TStatCollector;

        virtual void OnBegin();
        virtual void OnSuccess();
        virtual void OnCancel();
        virtual void OnFail();

        static TAtomic MaxContinuousErrors_;
        static TAtomic ReSendValidatorPeriod_;
        TAtomicCounter RequestsInProcess_;
        TAtomic LastContinuousErrors_ = 0;
        TAtomic SendValidatorCounter_ = 0;
    };

    using TServiceStatRef = TIntrusivePtr<TServiceStat>;

    //thread safe (race protected) service stat updater
    class TStatCollector {
    public:
        TStatCollector(TServiceStatRef& ss)
            : SS_(ss)
        {
            ss->OnBegin();
        }

        ~TStatCollector() {
            if (CanInformSS()) {
                SS_->OnFail();
            }
        }

        void OnCancel() noexcept {
            if (CanInformSS()) {
                SS_->OnCancel();
            }
        }

        void OnFail() noexcept {
            if (CanInformSS()) {
                SS_->OnFail();
            }
        }

        void OnSuccess() noexcept {
            if (CanInformSS()) {
                SS_->OnSuccess();
            }
        }

    private:
        inline bool CanInformSS() noexcept {
            return AtomicGet(CanInformSS_) && AtomicCas(&CanInformSS_, 0, 1);
        }

        TServiceStatRef SS_;
        TAtomic CanInformSS_ = 1;
    };

    TServiceStatRef GetServiceStat(TStringBuf addr);

}
