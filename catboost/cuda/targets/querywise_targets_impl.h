#pragma once

#include "target_func.h"
#include "kernel.h"
#include "oracle_type.h"
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/libs/metrics/pfound.h>
#include <catboost/cuda/gpu_data/dataset_base.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/cuda/cuda_util/transform.h>

namespace NCatboostCuda {
    template <class TDocLayout>
    class TQuerywiseTargetsImpl: public TQuerywiseTarget<TDocLayout> {
    public:
        using TParent = TQuerywiseTarget<TDocLayout>;
        using TStat = TAdditiveStatistic;
        using TMapping = TDocLayout;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        template <class TDataSet>
        TQuerywiseTargetsImpl(const TDataSet& dataSet,
                              TGpuAwareRandom& random,
                              TSlice slice,
                              const NCatboostOptions::TLossDescription& targetOptions,
                              const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor = Nothing())
            : TParent(dataSet,
                      random,
                      slice) {
            CB_ENSURE(!objectiveDescriptor.Defined(), "Querywise target is not supported for custom loss");
            Init(targetOptions);
        }

        template <class TDataSet>
        TQuerywiseTargetsImpl(const TDataSet& dataSet,
                              TGpuAwareRandom& random,
                              const NCatboostOptions::TLossDescription& targetOptions,
                              const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor = Nothing())
            : TParent(dataSet,
                      random)
        {
            CB_ENSURE(!objectiveDescriptor.Defined(), "Querywise target is not supported for custom loss");
            Init(targetOptions);
        }

        TQuerywiseTargetsImpl(const TQuerywiseTargetsImpl& target,
                              const TSlice& slice)
            : TParent(target, slice)
            , Params(target.GetParams())
            , ScoreMetric(target.GetScoreMetricType())
            , PairsTotalWeight(target.GetPairsTotalWeight())
            , TotalWeightedTarget(target.GetTotalWeightedTarget())
        {
        }

        TQuerywiseTargetsImpl(const TQuerywiseTargetsImpl& target)
            : TParent(target)
            , Params(target.GetParams())
            , ScoreMetric(target.GetScoreMetricType())
            , PairsTotalWeight(target.GetPairsTotalWeight())
            , TotalWeightedTarget(target.GetTotalWeightedTarget())
        {
        }

        template <class TLayout>
        TQuerywiseTargetsImpl(const TQuerywiseTargetsImpl<TLayout>& basedOn,
                              TTarget<TMapping>&& target)
            : TParent(basedOn, std::move(target))
            , Params(basedOn.GetParams())
        {
            Init(basedOn.GetParams());
        }

        TQuerywiseTargetsImpl(TQuerywiseTargetsImpl&& other)
            : TParent(std::move(other))
            , Params(other.Params)
            , ScoreMetric(other.GetScoreMetricType())
            , PairsTotalWeight(other.GetPairsTotalWeight())
            , TotalWeightedTarget(other.GetTotalWeightedTarget())
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;

        TAdditiveStatistic ComputeStats(const TConstVec& point,
                                        const TMap<TString, TString> params = TMap<TString, TString>()) const {
            double weight = 0;
            switch (ScoreMetric) {
                case ELossFunction::QueryRMSE: {
                    weight = GetTotalWeight();
                    break;
                }
                case ELossFunction::PairLogit: {
                    weight = PairsTotalWeight;
                    break;
                }
                case ELossFunction::QuerySoftMax: {
                    weight = TotalWeightedTarget;
                    break;
                }
                default: {
                    CB_ENSURE(false, "Unimplemented " << ScoreMetric);
                }
            }

            Y_UNUSED(params);
            TVector<float> result;
            auto tmp = TVec::Create(point.GetMapping().RepeatOnAllDevices(1));
            FillBuffer(tmp, 0.0f);

            ApproximateForPermutation(point,
                                      /*indices*/ nullptr,
                                      &tmp,
                                      nullptr,
                                      0,
                                      nullptr);

            NCudaLib::TCudaBufferReader<TVec>(tmp)
                .SetFactorSlice(TSlice(0, 1))
                .SetReadSlice(TSlice(0, 1))
                .ReadReduce(result);

            return MakeSimpleAdditiveStatistic(-result[0], weight);
        }

        void GradientAt(const TConstVec& point,
                        TVec& weightedDer,
                        TVec& weights,
                        ui32 stream = 0) const {
            if (Params.GetLossFunction() == ELossFunction::YetiRank) {
                NewtonAt(point, weightedDer, weights, stream);
            } else {
                ApproximateForPermutation(point,
                                          nullptr,
                                          nullptr,
                                          &weightedDer,
                                          0,
                                          nullptr,
                                          stream);
                weights.Copy(GetTarget().GetWeights(), stream);
            }
        }

        //For YetiRank Newton approximation is meaningless
        void NewtonAt(const TConstVec& point,
                      TVec& weightedDer,
                      TVec& weights,
                      ui32 stream = 0) const {
            ApproximateForPermutation(point,
                                      nullptr,
                                      nullptr,
                                      &weightedDer,
                                      0,
                                      &weights,
                                      stream);
        }

        /* point is not gathered for indices */
        void StochasticDer(const TConstVec& point,
                           const TVec& sampledWeights,
                           TBuffer<ui32>&& sampledIndices,
                           bool secondDerAsWeights,
                           TOptimizationTarget* target) const {
            auto der = TVec::CopyMapping(point);
            auto der2OrWeights = TVec::CopyMapping(point);

            if (secondDerAsWeights) {
                GradientAt(point, der, der2OrWeights);
            } else {
                NewtonAt(point, der, der2OrWeights);
            }

            target->StatsToAggregate.Reset(sampledWeights.GetMapping(), 2);
            auto weightsColumn = target->StatsToAggregate.ColumnView(0);
            auto derColumn = target->StatsToAggregate.ColumnView(1);
            Gather(weightsColumn, der2OrWeights, sampledIndices);
            Gather(derColumn, der, sampledIndices);

            MultiplyVector(weightsColumn, sampledWeights);
            MultiplyVector(derColumn, sampledWeights);
            target->Indices = std::move(sampledIndices);
        }

        void ApproximateForPermutation(const TConstVec& point,
                                       const TBuffer<ui32>* indices,
                                       TVec* value,
                                       TVec* der,
                                       ui32 der2Row,
                                       TVec* der2,
                                       ui32 stream = 0) const {
            CB_ENSURE(der2Row == 0, "This is single-dim target");
            const auto& samplesGrouping = TParent::GetSamplesGrouping();

            ELossFunction lossFunction = Params.GetLossFunction();
            switch (lossFunction) {
                case ELossFunction::QueryRMSE: {
                    ApproximateQueryRmse(samplesGrouping.GetSizes(),
                                         samplesGrouping.GetBiasedOffsets(),
                                         samplesGrouping.GetOffsetsBias(),
                                         GetTarget().GetTargets(),
                                         GetTarget().GetWeights(),
                                         point,
                                         indices,
                                         value,
                                         der,
                                         der2,
                                         stream);
                    break;
                }
                case ELossFunction::YetiRank: {
                    ApproximateYetiRank(TParent::GetRandom().NextUniformL(),
                                        NCatboostOptions::GetYetiRankDecay(Params),
                                        NCatboostOptions::GetYetiRankPermutations(Params),
                                        samplesGrouping.GetSizes(),
                                        samplesGrouping.GetBiasedOffsets(),
                                        samplesGrouping.GetOffsetsBias(),
                                        GetTarget().GetTargets(),
                                        GetTarget().GetWeights(),
                                        point,
                                        indices,
                                        value,
                                        der,
                                        der2,
                                        stream);
                    break;
                }
                case ELossFunction::PairLogit: {
                    ApproximatePairLogit(samplesGrouping.GetPairs(),
                                         samplesGrouping.GetPairsWeights(),
                                         samplesGrouping.GetOffsetsBias(),
                                         point,
                                         indices,
                                         value,
                                         der,
                                         der2,
                                         stream);
                    break;
                }
                case ELossFunction::QuerySoftMax: {
                    ApproximateQuerySoftMax(samplesGrouping.GetSizes(),
                                            samplesGrouping.GetBiasedOffsets(),
                                            samplesGrouping.GetOffsetsBias(),
                                            NCatboostOptions::GetQuerySoftMaxLambdaReg(Params),
                                            NCatboostOptions::GetQuerySoftMaxBeta(Params),
                                            GetTarget().GetTargets(),
                                            GetTarget().GetWeights(),
                                            point,
                                            indices,
                                            value,
                                            der,
                                            der2,
                                            stream);
                    break;
                }
                default: {
                    CB_ENSURE(false, "Unsupported querywise loss " << lossFunction);
                }
            }
        }

        bool IsMinOptimal() {
            switch (Params.GetLossFunction()) {
                case ELossFunction::YetiRank: {
                    return false;
                }
                case ELossFunction::QueryRMSE:
                case ELossFunction::PairLogit:
                case ELossFunction::QuerySoftMax: {
                    return true;
                }
                default: {
                    CB_ENSURE(false, "Unknown loss " << Params.GetLossFunction());
                }
            }
        }

        ELossFunction GetScoreMetricType() const {
            return ScoreMetric;
        }

        const NCatboostOptions::TLossDescription& GetParams() const {
            return Params;
        };

        ui32 GetDim() const {
            return 1;
        }

        ELossFunction GetType() const {
            return Params.GetLossFunction();
        }

        EHessianType GetHessianType() const {
            return EHessianType::Symmetric;
        }

        static constexpr EOracleType OracleType() {
            return EOracleType::Pointwise;
        }

        double GetPairsTotalWeight() const {
            return PairsTotalWeight;
        }

        double GetTotalWeightedTarget() const {
            return TotalWeightedTarget;
        }

    private:
        void InitYetiRank(const NCatboostOptions::TLossDescription& targetOptions) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::YetiRank);

            const auto& grouping = TParent::GetSamplesGrouping();
            for (ui32 qid = 0; qid < grouping.GetQueryCount(); ++qid) {
                const auto querySize = grouping.GetQuerySize(qid);
                CB_ENSURE(querySize <= 1023, "Error: max query size supported on GPU is 1023, got " << querySize);
            }
            ScoreMetric = ELossFunction::PFound;
        }

        void InitPairLogit(const NCatboostOptions::TLossDescription& targetOptions) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::PairLogit);
            TVec weights = TVec::CopyMapping(TParent::GetTarget().GetTargets());
            FillBuffer(weights, 0.0f);
            MakePairWeights(TParent::GetSamplesGrouping().GetPairs(),
                            TParent::GetSamplesGrouping().GetPairsWeights(),
                            weights);

            TParent::Target.Weights = weights.ConstCopyView();
            TParent::Target.StorePairWeights = true;

            {
                const auto& pairWeights = TParent::GetSamplesGrouping().GetPairsWeights();
                auto tmp = TVec::CopyMapping(pairWeights);
                FillBuffer(tmp, 1.0f);
                PairsTotalWeight = DotProduct(tmp, pairWeights);
                if (PairsTotalWeight <= 0) {
                    ythrow yexception() << "Observation weights should be greater or equal zero. Total weight should be greater, than zero";
                }
            }
        }

        void InitQuerySoftmax() {
            if (TotalWeightedTarget <= 0) {
                TotalWeightedTarget = DotProduct(GetTarget().GetTargets(),
                                                 GetTarget().GetWeights());
                if (TotalWeightedTarget <= 0) {
                    ythrow TCatBoostException() << "Observation targets and weights should be greater or equal zero. Total weighted target should be greater, than zero";
                }
            }
        }

        void Init(const NCatboostOptions::TLossDescription& options) {
            Params = options;
            ScoreMetric = options.GetLossFunction();
            auto objective = options.GetLossFunction();
            switch (objective) {
                case ELossFunction::YetiRank: {
                    InitYetiRank(Params);
                    break;
                }
                case ELossFunction::PairLogit: {
                    InitPairLogit(Params);
                    break;
                }
                case ELossFunction::QuerySoftMax: {
                    InitQuerySoftmax();
                    break;
                }
                case ELossFunction::QueryRMSE: {
                    break;
                }
                default: {
                    CB_ENSURE(false, "Unsupported querywise objective " << objective);
                }
            }
        }

    private:
        NCatboostOptions::TLossDescription Params;
        ELossFunction ScoreMetric;

        double PairsTotalWeight = 0;
        double TotalWeightedTarget = 0;
    };

}
