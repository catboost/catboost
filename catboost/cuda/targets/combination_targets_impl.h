#pragma once

#include "target_func.h"
#include "kernel.h"
#include "oracle_type.h"
#include "pointwise_target_impl.h"
#include "querywise_targets_impl.h"
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/pfound.h>
#include <catboost/cuda/gpu_data/dataset_base.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/cuda/cuda_util/transform.h>

namespace NCatboostCuda {
    inline bool IsDiagQuerywiseLoss(ELossFunction loss) {
        return EqualToOneOf(
            loss,
            ELossFunction::YetiRank,
            ELossFunction::PairLogit,
            ELossFunction::QuerySoftMax,
            ELossFunction::QueryRMSE);
    }

    struct TWeightAndLoss {
        float Weight;
        NCatboostOptions::TLossDescription Loss;
    };

    template <class TDocLayout>
    class TCombinationTargetsImpl: public TQuerywiseTarget<TDocLayout> {
    public:
        using TParent = TQuerywiseTarget<TDocLayout>;
        using TStat = TAdditiveStatistic;
        using TMapping = TDocLayout;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        template <class TDataSet>
        TCombinationTargetsImpl(
            const TDataSet& dataSet,
            TGpuAwareRandom& random,
            TSlice slice,
            const NCatboostOptions::TLossDescription& targetOptions,
            const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor = Nothing())
        : TParent(dataSet, random, slice)
        {
            CB_ENSURE(!objectiveDescriptor.Defined());
            CreateLosses(targetOptions);

            const auto createQuerywiseTarget = [&] (const auto& weightAndLoss) {
                return MakeHolder<TQuerywiseTargetsImpl<TDocLayout>>(dataSet, random, slice, weightAndLoss.Loss);
            };
            CreateElementwise(createQuerywiseTarget, QuerywiseLosses, &QuerywiseTargets);

            const auto createPointwiseTarget = [&] (const auto& weightAndLoss) {
                return MakeHolder<TPointwiseTargetsImpl<TDocLayout>>(dataSet, random, slice, weightAndLoss.Loss);
            };
            CreateElementwise(createPointwiseTarget, PointwiseLosses, &PointwiseTargets);
        }

        template <class TDataSet>
        TCombinationTargetsImpl(
            const TDataSet& dataSet,
            TGpuAwareRandom& random,
            const NCatboostOptions::TLossDescription& targetOptions,
            const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor = Nothing())
        : TParent(dataSet, random)
        {
            CB_ENSURE(!objectiveDescriptor.Defined());
            CreateLosses(targetOptions);

            const auto createQuerywiseTarget = [&] (const auto& weightAndLoss) {
                return MakeHolder<TQuerywiseTargetsImpl<TDocLayout>>(dataSet, random, weightAndLoss.Loss);
            };
            CreateElementwise(createQuerywiseTarget, QuerywiseLosses, &QuerywiseTargets);

            const auto createPointwiseTarget = [&] (const auto& weightAndLoss) {
                return MakeHolder<TPointwiseTargetsImpl<TDocLayout>>(dataSet, random, weightAndLoss.Loss);
            };
            CreateElementwise(createPointwiseTarget, PointwiseLosses, &PointwiseTargets);
        }

        TCombinationTargetsImpl(
            const TCombinationTargetsImpl& target,
            const TSlice& slice)
        : TParent(target, slice)
        , QuerywiseLosses(target.QuerywiseLosses)
        , PointwiseLosses(target.PointwiseLosses)
        {
            const auto createQuerywiseTarget = [&] (const auto& target) {
                return MakeHolder<TQuerywiseTargetsImpl<TDocLayout>>(*target, slice);
            };
            CreateElementwise(createQuerywiseTarget, target.QuerywiseTargets, &QuerywiseTargets);

            const auto createPointwiseTarget = [&] (const auto& target) {
                return MakeHolder<TPointwiseTargetsImpl<TDocLayout>>(*target, slice);
            };
            CreateElementwise(createPointwiseTarget, target.PointwiseTargets, &PointwiseTargets);
        }

        TCombinationTargetsImpl(const TCombinationTargetsImpl& target)
        : TParent(target)
        , QuerywiseLosses(target.QuerywiseLosses)
        , PointwiseLosses(target.PointwiseLosses)
        {
            const auto createQuerywiseTarget = [&] (const auto& target) {
                return MakeHolder<TQuerywiseTargetsImpl<TDocLayout>>(*target);
            };
            CreateElementwise(createQuerywiseTarget, target.QuerywiseTargets, &QuerywiseTargets);

            const auto createPointwiseTarget = [&] (const auto& target) {
                return MakeHolder<TPointwiseTargetsImpl<TDocLayout>>(*target);
            };
            CreateElementwise(createPointwiseTarget, target.PointwiseTargets, &PointwiseTargets);
        }

        TCombinationTargetsImpl(
            const TCombinationTargetsImpl<NCudaLib::TMirrorMapping>& basedOn,
            TTarget<TDocLayout>&& target)
        : TParent(basedOn, std::move(TTarget<TDocLayout>(target)))
        , QuerywiseLosses(basedOn.QuerywiseLosses)
        , PointwiseLosses(basedOn.PointwiseLosses)
        {
            const auto createQuerywiseTarget = [&] (const auto& basedOn) {
                return MakeHolder<TQuerywiseTargetsImpl<TDocLayout>>(*basedOn, std::move(TTarget<TDocLayout>(target)));
            };
            CreateElementwise(createQuerywiseTarget, basedOn.QuerywiseTargets, &QuerywiseTargets);

            const auto createPointwiseTarget = [&] (const auto& basedOn) {
                return MakeHolder<TPointwiseTargetsImpl<TDocLayout>>(*basedOn, std::move(TTarget<TDocLayout>(target)));
            };
            CreateElementwise(createPointwiseTarget, basedOn.PointwiseTargets, &PointwiseTargets);
        }

        TCombinationTargetsImpl(TCombinationTargetsImpl&& other)
        : TParent(other)
        , QuerywiseLosses(other.QuerywiseLosses)
        , PointwiseLosses(other.PointwiseLosses)
        {
            const auto createQuerywiseTarget = [&] (const auto& target) {
                return MakeHolder<TQuerywiseTargetsImpl<TDocLayout>>(*target);
            };
            CreateElementwise(createQuerywiseTarget, other.QuerywiseTargets, &QuerywiseTargets);

            const auto createPointwiseTarget = [&] (const auto& target) {
                return MakeHolder<TPointwiseTargetsImpl<TDocLayout>>(*target);
            };
            CreateElementwise(createPointwiseTarget, other.PointwiseTargets, &PointwiseTargets);
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;

        TAdditiveStatistic ComputeStats(
            const TConstVec& point,
            const TMap<TString, TString> params = TMap<TString, TString>()
        ) const {
            Y_UNUSED(params);
            TAdditiveStatistic totalStats;
            for (ui32 idx : xrange(QuerywiseLosses.size())) {
                const auto& metrics = CreateMetricFromDescription(QuerywiseLosses[idx].Loss, /*approxDimension*/ 1);
                const auto& stats = QuerywiseTargets[idx]->ComputeStats(point, QuerywiseLosses[idx].Loss.GetLossParamsMap());
                const double value = metrics[0]->GetFinalError(stats);
                totalStats.Add(MakeSimpleAdditiveStatistic(value * QuerywiseLosses[idx].Weight, 0));
            }
            for (ui32 idx : xrange(PointwiseLosses.size())) {
                const auto& metrics = CreateMetricFromDescription(PointwiseLosses[idx].Loss, /*approxDimension*/ 1);
                const auto& stats = PointwiseTargets[idx]->ComputeStats(point, PointwiseLosses[idx].Loss.GetLossParamsMap());
                const double value = metrics[0]->GetFinalError(stats);
                totalStats.Add(MakeSimpleAdditiveStatistic(value * PointwiseLosses[idx].Weight, 0));
            }
            return totalStats;
        }

        void GradientAt(
            const TConstVec& point,
            TVec& weightedDer,
            TVec& weights,
            ui32 stream = 0
        ) const {
            auto scratchWeightedDer = TVec::CopyMapping(point);
            auto scratchWeights = TVec::CopyMapping(point);
            FillBuffer(weightedDer, 0.0f, stream);
            FillBuffer(weights, 0.0f, stream);
            for (ui32 idx : xrange(QuerywiseLosses.size())) {
                QuerywiseTargets[idx]->GradientAt(point, scratchWeightedDer, scratchWeights, stream);
                const float weight = QuerywiseLosses[idx].Weight;
                MultiplyAddVector(weightedDer, scratchWeightedDer, weight, stream);
                MultiplyAddVector(weights, scratchWeights, weight, stream);
            }
            for (ui32 idx : xrange(PointwiseLosses.size())) {
                PointwiseTargets[idx]->GradientAt(point, scratchWeightedDer, scratchWeights, stream);
                const float weight = PointwiseLosses[idx].Weight;
                MultiplyAddVector(weightedDer, scratchWeightedDer, weight, stream);
                MultiplyAddVector(weights, scratchWeights, weight, stream);
            }
        }

        void NewtonAt(
            const TConstVec& point,
            TVec& weightedDer,
            TVec& weights,
            ui32 stream = 0
        ) const {
            auto scratchWeightedDer = TVec::CopyMapping(weightedDer);
            auto scratchWeights = TVec::CopyMapping(weights);
            FillBuffer(weightedDer, 0.0f, stream);
            FillBuffer(weights, 0.0f, stream);
            for (ui32 idx : xrange(QuerywiseLosses.size())) {
                QuerywiseTargets[idx]->NewtonAt(point, scratchWeightedDer, scratchWeights, stream);
                const float weight = QuerywiseLosses[idx].Weight;
                MultiplyAddVector(weightedDer, scratchWeightedDer, weight, stream);
                MultiplyAddVector(weights, scratchWeights, weight, stream);
            }
            for (ui32 idx : xrange(PointwiseLosses.size())) {
                PointwiseTargets[idx]->NewtonAt(point, scratchWeightedDer, scratchWeights, stream);
                const float weight = PointwiseLosses[idx].Weight;
                MultiplyAddVector(weightedDer, scratchWeightedDer, weight, stream);
                MultiplyAddVector(weights, scratchWeights, weight, stream);
            }
        }

        /* point is not gathered for indices */
        void StochasticDer(
            const TConstVec& point,
            const TVec& sampledWeights,
            const TBuffer<ui32>& sampledIndices,
            bool secondDerAsWeights,
            TOptimizationTarget* target
        ) const {
            CB_ENSURE(point.GetColumnCount() == 1, "Unimplemented for multidim targets");
            auto totalWeightedDer = TVec::CopyMapping(sampledWeights);
            auto totalWeights = TVec::CopyMapping(sampledWeights);
            FillBuffer(totalWeightedDer, 0.0f);
            FillBuffer(totalWeights, 0.0f);
            for (ui32 idx : xrange(QuerywiseLosses.size())) {
                QuerywiseTargets[idx]->StochasticDer(point, sampledWeights, sampledIndices, secondDerAsWeights, target);
                CB_ENSURE(target->StatsToAggregate.GetColumnCount() == 2, "Expect exactly two stats to aggregate in combination target");
                const float weight = QuerywiseLosses[idx].Weight;

                auto weights = target->StatsToAggregate.ColumnView(0);
                MultiplyAddVector(totalWeights, weights, weight);

                auto der = target->StatsToAggregate.ColumnView(1);
                MultiplyAddVector(totalWeightedDer, der, weight);
            }
            for (ui32 idx : xrange(PointwiseLosses.size())) {
                PointwiseTargets[idx]->StochasticDer(point, sampledWeights, sampledIndices, secondDerAsWeights, target);
                CB_ENSURE(target->StatsToAggregate.GetColumnCount() == 2, "Expect exactly two stats to aggregate");
                const float weight = PointwiseLosses[idx].Weight;

                auto weights = target->StatsToAggregate.ColumnView(0);
                MultiplyAddVector(totalWeights, weights, weight);

                auto der = target->StatsToAggregate.ColumnView(1);
                MultiplyAddVector(totalWeightedDer, der, weight);
            }

            target->StatsToAggregate.Reset(sampledWeights.GetMapping(), 2);
            target->StatsToAggregate.ColumnView(0).Copy(totalWeights);
            target->StatsToAggregate.ColumnView(1).Copy(totalWeightedDer);
            target->Indices = std::move(sampledIndices);
        }

        void ApproximateForPermutation(
            const TConstVec& point,
            const TBuffer<ui32>* indices,
            TVec* value,
            TVec* der,
            ui32 der2Row,
            TVec* der2,
            ui32 stream = 0
        ) const {
            CB_ENSURE(point.GetColumnCount() == 1, "Unimplemented for loss with multiple columns");

            auto scratchValue = TVec::CopyMapping(indices);
            auto scratchDer = TVec::CopyMapping(indices);
            TCudaBuffer<float, TDocLayout> scratchDer2;
            if (der2 != nullptr) {
                scratchDer2 = TVec::CopyMapping(indices);
            }

            FillBuffer(*value, 0.0f, stream);
            FillBuffer(*der, 0.0f, stream);
            if (der2) {
                FillBuffer(*der2, 0.0f, stream);
            }
            for (ui32 idx : xrange(QuerywiseLosses.size())) {
                QuerywiseTargets[idx]->ApproximateForPermutation(
                    point.AsConstBuf(),
                    indices,
                    &scratchValue,
                    &scratchDer,
                    der2Row,
                    &scratchDer2,
                    stream
                );
                const float weight = QuerywiseLosses[idx].Weight;
                MultiplyAddVector(*value, scratchValue, weight, stream);
                MultiplyAddVector(*der, scratchDer, weight, stream);
                if (der2 != nullptr) {
                    MultiplyAddVector(*der2, scratchDer2, weight, stream);
                }
            }

            auto inverseIndices = TBuffer<ui32>::CopyMapping(indices);
            InversePermutation(*indices, inverseIndices, stream);
            auto target = TVec::CopyMapping(indices);
            Gather(target, GetTarget().GetTargets(), inverseIndices, stream);
            auto weights = TVec::CopyMapping(indices);
            Gather(weights, GetTarget().GetWeights(), inverseIndices, stream);

            for (ui32 idx : xrange(PointwiseLosses.size())) {
                PointwiseTargets[idx]->Approximate(
                    target.AsConstBuf(),
                    weights.AsConstBuf(),
                    point.AsConstBuf(),
                    &scratchValue,
                    &scratchDer,
                    0,
                    &scratchDer2,
                    stream
                );
                const float weight = PointwiseLosses[idx].Weight;
                MultiplyAddVector(*value, scratchValue, weight, stream);
                MultiplyAddVector(*der, scratchDer, weight, stream);
                if (der2 != nullptr) {
                    MultiplyAddVector(*der2, scratchDer2, weight, stream);
                }
            }
        }

        static constexpr bool IsMinOptimal() {
            return true;
        }

        static constexpr ELossFunction GetScoreMetricType() {
            return ELossFunction::Combination;
        }

        static constexpr ui32 GetDim() {
            return 1;
        }

        static constexpr ELossFunction GetType() {
            return ELossFunction::Combination;
        }

        static constexpr EHessianType GetHessianType() {
            return EHessianType::Symmetric;
        }

        static constexpr EOracleType OracleType() {
            return EOracleType::Pointwise;
        }

    private:
        template <typename TCreator, typename TParameter, typename TCreature>
        static void CreateElementwise(const TCreator creator, const TVector<TParameter>& parameters, TVector<TCreature>* creatures) {
            creatures->clear();
            for (const auto& parameter : parameters) {
                creatures->emplace_back(creator(parameter));
            }
        }

        void CreateLosses(const NCatboostOptions::TLossDescription& targetOptions) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::Combination,  "Only combination loss is supported");
            IterateOverCombination(
                targetOptions.GetLossParamsMap(),
                [&] (const auto& loss, float weight) {
                    const auto lossFunction = loss.GetLossFunction();
                    if (IsDiagQuerywiseLoss(lossFunction)) {
                        if (lossFunction == ELossFunction::YetiRank) {
                            weight = - weight;
                        }
                        QuerywiseLosses.emplace_back(TWeightAndLoss{weight, loss});
                    } else {
                        PointwiseLosses.emplace_back(TWeightAndLoss{weight, loss});
                    }
            });
        }

    public:
        TVector<TWeightAndLoss> QuerywiseLosses;
        TVector<TWeightAndLoss> PointwiseLosses;
        TVector<THolder<TQuerywiseTargetsImpl<TDocLayout>>> QuerywiseTargets;
        TVector<THolder<TPointwiseTargetsImpl<TDocLayout>>> PointwiseTargets;
    };

}
