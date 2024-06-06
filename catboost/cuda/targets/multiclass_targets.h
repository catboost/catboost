#pragma once

#include "oracle_type.h"
#include "target_func.h"

#include <catboost/cuda/cuda_lib/mapping.h>

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>

#include <util/generic/algorithm.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/strbuf.h>
#include <util/string/cast.h>
#include <util/system/yassert.h>

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
                                    const NCatboostOptions::TLossDescription& targetOptions,
                                    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor)
            : TParent(dataSet,
                      random) {
            CB_ENSURE(!objectiveDescriptor.Defined());
            Init(targetOptions, dataSet.GetDataProvider());
        }

        TMultiClassificationTargets(TMultiClassificationTargets&& other)
            : TParent(std::move(other))
            , Type(other.GetType())
            , MetricName(other.ScoreMetricName())
            , NumClasses(other.NumClasses)
        {
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

        EHessianType GetHessianType() const {
            if (Type == ELossFunction::MultiClassOneVsAll) {
                return EHessianType::Diagonal;
            }
            return EHessianType::Symmetric;
        }

        ELossFunction GetScoreMetricType() const {
            return Type;
        }

        ui32 GetDim() const {
            if (Type == ELossFunction::MultiClass) {
                return NumClasses - 1;
            } else {
                return NumClasses;
            }
        }

        static constexpr EOracleType OracleType() {
            return EOracleType::Pointwise;
        }

    private:
        void Init(const NCatboostOptions::TLossDescription& targetOptions,
                  const NCB::TTrainingDataProvider& dataProvider) {
            auto targetClassCount = dataProvider.TargetData->GetTargetClassCount();
            if (targetClassCount) {
                NumClasses = *targetClassCount;
                if (!IsMultiLabelObjective(targetOptions.GetLossFunction())) {
                    TConstArrayRef<float> target = *dataProvider.TargetData->GetOneDimensionalTarget();
                    TVector<float> tmp(target.begin(), target.end());
                    SortUnique(tmp);
                    CB_ENSURE(
                        NumClasses >= tmp.size(),
                        "Number of classes (" << NumClasses << ") should be >= number of unique labels (" << tmp.size() << ")");
                }
            } else if (targetOptions.GetLossFunction() == ELossFunction::MultiRMSE) {
                NumClasses = dataProvider.TargetData->GetTargetDimension();
            } else {
                CB_ENSURE_INTERNAL(targetOptions.GetLossFunction() == ELossFunction::RMSEWithUncertainty,
                                "dataProvider.TargetData must contain class count, or loss function must be RMSEWithUncertainty");
                NumClasses = 2;
            }

            CATBOOST_DEBUG_LOG << "Num classes " << NumClasses << Endl;
            Type = targetOptions.GetLossFunction();
            MetricName = ToString(targetOptions);

            CB_ENSURE(NumClasses > 1, "Only one class found, can't learn multiclass objective");
        }

    private:
        ELossFunction Type = ELossFunction::PythonUserDefinedPerObject;
        TString MetricName;
        ui32 NumClasses = 0;
    };

}
