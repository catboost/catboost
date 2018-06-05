#pragma once

#include "target_func.h"
#include "kernel.h"
#include <catboost/cuda/targets/kernel/pointwise_targets.cuh>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/metrics/metric.h>

namespace NCatboostCuda {


    template <class TDocLayout, class TDataSet>
    class TCrossEntropy: public TPointwiseTarget<TDocLayout, TDataSet> {
    public:
        using TParent = TPointwiseTarget<TDocLayout, TDataSet>;
        using TStat = TAdditiveStatistic;
        using TMapping = TDocLayout;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        virtual ~TCrossEntropy() {
        }

        TCrossEntropy(const TDataSet& dataSet,
                      TGpuAwareRandom& random,
                      TSlice slice,
                      const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random,
                      slice) {
            Y_UNUSED(targetOptions);
        }

        TCrossEntropy(const TDataSet& dataSet,
                      TGpuAwareRandom& random,
                      const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random) {
            Y_UNUSED(targetOptions);
        }

        TCrossEntropy(const TCrossEntropy& target,
                      const TSlice& slice)
            : TParent(target,
                      slice) {
        }

        TCrossEntropy(const TCrossEntropy& target)
            : TParent(target)
        {
        }

        template <class TLayout>
        TCrossEntropy(const TCrossEntropy<TLayout, TDataSet>& basedOn,
                      TTarget<TMapping>&& target)
            : TParent(basedOn.GetDataSet(),
                      basedOn.GetRandom(),
                      std::move(target)) {
        }

        TCrossEntropy(TCrossEntropy&& other)
            : TParent(std::move(other))
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;

        TAdditiveStatistic ComputeStats(const TConstVec& point,
                                        const TMap<TString, TString> params = TMap<TString, TString>()) const {
            Y_UNUSED(params);
            TVector<float> result;
            auto tmp = TVec::Create(point.GetMapping().RepeatOnAllDevices(1));

            Approximate(GetTarget().GetTargets(),
                        GetTarget().GetWeights(),
                        point,
                        &tmp,
                        nullptr,
                        nullptr);

            NCudaLib::TCudaBufferReader<TVec>(tmp)
                .SetFactorSlice(TSlice(0, 1))
                .SetReadSlice(TSlice(0, 1))
                .ReadReduce(result);

            const double weight = GetTotalWeight();

            return MakeSimpleAdditiveStatistic(result[0], weight);
        }


        static double Score(const TAdditiveStatistic& score) {
            return -score.Stats[0] / score.Stats[1];
        }

        double Score(const TConstVec& point) const {
            return Score(ComputeStats(point));
        }

        void GradientAt(const TConstVec& point,
                        TVec& weightedDer,
                        TVec& weights,
                        ui32 stream = 0) const {
            Approximate(GetTarget().GetTargets(),
                        GetTarget().GetWeights(),
                        point,
                        nullptr,
                        &weightedDer,
                        nullptr,
                        stream);
            weights.Copy(GetTarget().GetWeights(), stream);
        }

        void NewtonAt(const TConstVec& point,
                      TVec& weightedDer,
                      TVec& weightedDer2,
                      ui32 stream = 0) const {
            Approximate(GetTarget().GetTargets(),
                        GetTarget().GetWeights(),
                        point,
                        nullptr,
                        &weightedDer,
                        &weightedDer2,
                        stream);
            weightedDer2.Copy(GetTarget().GetWeights(), stream);
        }

        void Approximate(const TConstVec& target,
                         const TConstVec& weights,
                         const TConstVec& point,
                         TVec* value,
                         TVec* der,
                         TVec* der2,
                         ui32 stream = 0) const {
            ApproximateCrossEntropy(target,
                                    weights,
                                    point,
                                    value,
                                    der,
                                    der2,
                                    UseBorder(),
                                    GetBorder(),
                                    stream);
        }

        static constexpr TStringBuf ScoreMetricName() {
            return "CrossEntropy";
        }

        bool CouldComputeMetric(const IMetric& metric) const {
            return metric.GetDescription() == "CrossEntropy";
        }

        double ComputeMetric(const IMetric& metric,
                             const TConstVec& point) const {
            CB_ENSURE(CouldComputeMetric(metric), "Can't compute metric " << metric);
            return Score(point);
        }

        static constexpr bool IsMinOptimal() {
            return true;
        }

        virtual bool UseBorder() const {
            return false;
        }

        virtual double GetBorder() const {
            return 0;
        }

        ELossFunction GetScoreMetricType() const {
            return ELossFunction::QueryCrossEntropy;
        }
    };

    template <class TDocLayout, class TDataSet>
    class TLogloss: public TCrossEntropy<TDocLayout, TDataSet> {
    public:
        using TParent = TCrossEntropy<TDocLayout, TDataSet>;
        using TStat = TAdditiveStatistic;
        using TMapping = TDocLayout;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TLogloss(const TDataSet& dataSet,
                 TGpuAwareRandom& random,
                 TSlice slice,
                 const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random,
                      slice,
                      targetOptions)
            , Border(NCatboostOptions::GetLogLossBorder(targetOptions))
        {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::Logloss);
        }

        TLogloss(const TDataSet& dataSet,
                 TGpuAwareRandom& random,
                 const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random,
                      targetOptions)
            , Border(NCatboostOptions::GetLogLossBorder(targetOptions))
        {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::Logloss);
        }

        TLogloss(const TLogloss& target,
                 const TSlice& slice)
            : TParent(target,
                      slice)
            , Border(target.GetBorder())
        {
        }

        TLogloss(const TLogloss& target)
            : TParent(target)
            , Border(target.GetBorder())
        {
        }

        template <class TLayout>
        TLogloss(const TLogloss<TLayout, TDataSet>& basedOn,
                 TTarget<TMapping>&& target)
            : TParent(basedOn,
                      std::move(target))
            , Border(basedOn.GetBorder())
        {
        }

        TLogloss(TLogloss&& other)
            : TParent(std::move(other))
            , Border(other.GetBorder())
        {
        }

        TStringBuf ScoreMetricName() const {
            if (Border != 0.5) {
                return TStringBuilder() << "Logloss:Border=" << Border;
            }
            return "Logloss";
        }

        ELossFunction GetScoreMetricType() const {
            return ELossFunction::Logloss;
        }

        static constexpr bool IsMinOptimal() {
            return true;
        }

        bool UseBorder() const override {
            return true;
        }

        double GetBorder() const override {
            return Border;
        }

    private:
        double Border = 0.5;
    };
}
