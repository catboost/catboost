#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <util/generic/string.h>
#include <util/generic/ptr.h>

namespace NCatboostCuda {
    class IAdditiveStatistic {
    public:
        virtual ~IAdditiveStatistic() {
        }
    };

    struct TAdditiveStatistic: public IAdditiveStatistic {
        double Sum;
        double Weight;

        TAdditiveStatistic(double sum = 0.0,
                           double weight = 0.0)
            : Sum(sum)
            , Weight(weight)
        {
        }

        TAdditiveStatistic& operator+=(const TAdditiveStatistic& other) {
            Sum += other.Sum;
            Weight += other.Weight;
            return *this;
        }
    };

    template <class TTarget>
    class TMetricHelper {
    public:
        using TVec = typename TTarget::TVec;
        using TConstVec = typename TTarget::TConstVec;
        using TTargetStat = typename TTarget::TStat;

        explicit TMetricHelper(const TTarget& owner)
            : Owner(owner)
        {
        }

        TMetricHelper& SetPoint(const TConstVec& point) {
            CurrentStats = Owner.ComputeStats(point);
            return *this;
        }

        TString MetricName() const {
            return TString(Owner.TargetName());
        }

        TString ToTsv() const {
            return TStringBuilder() << MetricName() << "\t" << TTarget::Score(CurrentStats);
        }

        double Score() const {
            return TTarget::Score(CurrentStats);
        }

        double Score(const TTargetStat& stat) const {
            return TTarget::Score(stat);
        }

        TString TsvHeader() const {
            return TStringBuilder() << MetricName() << "\tScore";
        }

        const TTargetStat& GetStat() const {
            return CurrentStats;
        }

        bool IsBetter(const TTargetStat& other) {
            return (TTarget::Score(CurrentStats) < TTarget::Score(other)) == TTarget::IsMinOptimal();
        }

    private:
        const TTarget& Owner;
        TTargetStat CurrentStats;
    };
}
