#pragma once

#include "target_func.h"

#include "non_diag_target_der.h"
#include "oracle_type.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/gpu_data/querywise_helper.h>

#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/bootstrap_options.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>

#include <util/generic/ptr.h>
#include <util/generic/utility.h>
#include <util/system/compiler.h>

namespace NCatboostCuda {
    template <class TSamplesMapping>
    class TPFoundF;

    template <>
    class TPFoundF<NCudaLib::TStripeMapping>: public TNonDiagQuerywiseTarget<NCudaLib::TStripeMapping> {
    public:
        using TSamplesMapping = NCudaLib::TStripeMapping;
        using TParent = TNonDiagQuerywiseTarget<TSamplesMapping>;
        using TStat = TAdditiveStatistic;
        using TMapping = TSamplesMapping;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        template <class TDataSet>
        TPFoundF(const TDataSet& dataSet,
                 TGpuAwareRandom& random,
                 const NCatboostOptions::TLossDescription& targetOptions,
                 const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor)
            : TParent(dataSet,
                      random) {
            CB_ENSURE(!objectiveDescriptor.Defined());
            Init(targetOptions);
        }

        TPFoundF(TPFoundF&& other)
            : TParent(std::move(other))
            , PermutationCount(other.PermutationCount)
            , Decay(other.Decay)
        {
        }

        using TParent::GetTarget;

        TAdditiveStatistic ComputeStats(const TConstVec& point, const TMap<TString, TString> params = TMap<TString, TString>()) const {
            Y_UNUSED(point);
            Y_UNUSED(params);
            CB_ENSURE(false, "Unimplemented");
        }

        void StochasticGradient(const TConstVec& point,
                                const NCatboostOptions::TBootstrapConfig& config,
                                TNonDiagQuerywiseTargetDers* target) const {
            ApproximateStochastic(point, config, false, target);
        }

        void StochasticNewton(const TConstVec& point,
                              const NCatboostOptions::TBootstrapConfig& config,
                              TNonDiagQuerywiseTargetDers* target) const {
            ApproximateStochastic(point, config, true, target);
        }

        void ApproximateStochastic(const TConstVec& point,
                                   const NCatboostOptions::TBootstrapConfig& bootstrapConfig,
                                   bool,
                                   TNonDiagQuerywiseTargetDers* target) const;

        void FillPairsAndWeightsAtPoint(const TConstVec& point,
                                        TStripeBuffer<uint2>* pairs,
                                        TStripeBuffer<float>* pairWeights) const;

        void ApproximateAt(const TConstVec& point,
                           const TStripeBuffer<uint2>& pairs,
                           const TStripeBuffer<float>& pairWeights,
                           const TStripeBuffer<ui32>& scatterDerIndices,
                           TStripeBuffer<float>* value,
                           TStripeBuffer<float>* der,
                           TStripeBuffer<float>* pairDer2) const;

        static constexpr bool IsMinOptimal() {
            return false;
        }

        ELossFunction GetScoreMetricType() const {
            return ELossFunction::PFound;
        }

        ELossFunction GetType() const {
            return ELossFunction::YetiRankPairwise;
        }

        EHessianType GetHessianType() const {
            return EHessianType::Symmetric;
        }

        ui32 GetPFoundPermutationCount() const {
            return PermutationCount;
        }

        float GetDecay() const {
            return Decay;
        }

        static constexpr EOracleType OracleType() {
            return EOracleType::Pairwise;
        }

    private:
        ui32 GetMaxQuerySize() const {
            auto& queriesInfo = TParent::GetSamplesGrouping();
            const ui32 queryCount = queriesInfo.GetQueryCount();
            const double meanQuerySize = GetTarget().GetTargets().GetObjectsSlice().Size() * 1.0 / queryCount;
            const ui32 estimatedQuerySizeLimit = 2 * meanQuerySize + 8;
            return Min<ui32>(estimatedQuerySizeLimit, 1023);
        }

        void Init(const NCatboostOptions::TLossDescription& targetOptions) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::YetiRankPairwise);
            PermutationCount = NCatboostOptions::GetYetiRankPermutations(targetOptions);
            Decay = NCatboostOptions::GetYetiRankDecay(targetOptions);
        }

        TQuerywiseSampler& GetQueriesSampler() const {
            if (QueriesSampler == nullptr) {
                QueriesSampler = MakeHolder<TQuerywiseSampler>();
            }
            return *QueriesSampler;
        }

    private:
        mutable THolder<TQuerywiseSampler> QueriesSampler;
        ui32 PermutationCount = 10;
        float Decay = 0.99f;
    };

}
