#pragma once

#include "target_func.h"
#include "oracle_type.h"
#include <catboost/libs/options/loss_description.h>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>

namespace NCatboostCuda {

    template <class TSamplesMapping>
    class TMultiClassificationTargets;

    template <>
    class TMultiClassificationTargets<NCudaLib::TStripeMapping>: public TPointwiseTarget<NCudaLib::TStripeMapping> {
    public:
        using TParent = TPointwiseTarget<NCudaLib::TStripeMapping>;
        using TStat = TAdditiveStatistic;
        using TMapping = NCudaLib::TStripeMapping;
        CB_DEFINE_CUDA_TARGET_BUFFERS();


        template <class TDataSet>
        TMultiClassificationTargets(const TDataSet& dataSet,
                                    TGpuAwareRandom& random,
                                    const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random) {
            Init(targetOptions, dataSet.GetDataProvider());
        }


        TMultiClassificationTargets(TMultiClassificationTargets&& other)
            : TParent(std::move(other))
            , Type(other.GetType())
            , MetricName(other.ScoreMetricName())
            , NumClasses(other.NumClasses) {
        }

        TMultiClassificationTargets(const TMultiClassificationTargets& target)
                : TParent(target)
                  , Type(target.GetType())
                  , MetricName(target.ScoreMetricName())
                  , NumClasses(target.NumClasses)
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;

        TAdditiveStatistic ComputeStats(const TConstVec& point) const;

        TAdditiveStatistic ComputeStats(const TConstVec& point,
                                        const TMap<TString, TString>&) const {
            return ComputeStats(point);
        }

        double Score(const TAdditiveStatistic& score) const {
            return -score.Stats[0] / score.Stats[1];
        }

        double Score(const TConstVec& point) const {
            return Score(ComputeStats(point));
        }

        void Approximate(const TConstVec& target,
                         const TConstVec& weights,
                         const TConstVec& point,
                         TVec* value,
                         TVec* der,
                         ui32 der2Row,
                         TVec* der2,
                         ui32 stream = 0) const {
            if (value || der) {
                ComputeValueAndFirstDer(target, weights, point, value, der, stream);
            }
            if (der2) {
                ComputeSecondDerLine(target, weights, point, der2Row, der2, stream);
            }
        }

        void StochasticDer(const TConstVec& point,
                           const TVec& sampledWeights,
                           TBuffer<ui32>&& sampledIndices,
                           bool secondDerAsWeights,
                           TOptimizationTarget* target) const;

        void ComputeValueAndFirstDer(const TConstVec& target,
                                     const TConstVec& weights,
                                     const TConstVec& point,
                                     TVec* value,
                                     TVec* der,
                                     ui32 stream = 0) const;

        void ComputeSecondDerLine(const TConstVec& target,
                                  const TConstVec& weights,
                                  const TConstVec& point,
                                  ui32 row,
                                  TVec* der2Line,
                                  ui32 stream = 0) const;

        TStringBuf ScoreMetricName() const {
            return MetricName;
        }

        static constexpr bool IsMinOptimal() {
            return true;
        }

        ELossFunction GetType() const {
            return Type;
        }

        ELossFunction GetScoreMetricType() const {
            return Type;
        }

        ui32 GetDim() const {
            return NumClasses - 1;
        }

        static constexpr EOracleType OracleType() {
            return EOracleType::Pointwise;
        }
    private:
        void Init(const NCatboostOptions::TLossDescription& targetOptions,
                  const TDataProvider& dataProvider) {
            NumClasses = dataProvider.GetTargetHelper().GetNumClasses();
            TVector<float> tmp = dataProvider.GetTargets();
            SortUnique(tmp);
            Y_VERIFY(NumClasses >= tmp.size());
            MATRIXNET_DEBUG_LOG << "Num classes " << NumClasses << Endl;
            Type = targetOptions.GetLossFunction();
            MetricName = ToString(targetOptions);

            CB_ENSURE(Type == ELossFunction::MultiClass, Type);
            CB_ENSURE(NumClasses > 1, "Only one class found, can't learn multiclass objective");
        }
    private:
        ELossFunction Type = ELossFunction::Custom;
        TString MetricName;
        ui32 NumClasses = 0;
    };




}
