#pragma once

#include "target_func.h"

#include "non_diag_target_der.h"
#include "oracle_type.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/mapping.h>

#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/bootstrap_options.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>

#include <util/generic/map.h>
#include <util/generic/string.h>

namespace NCatboostCuda {
    template <class TSamplesMapping>
    class TPairLogitPairwise;

    template <>
    class TPairLogitPairwise<NCudaLib::TStripeMapping>: public TNonDiagQuerywiseTarget<NCudaLib::TStripeMapping> {
    public:
        using TSamplesMapping = NCudaLib::TStripeMapping;
        using TParent = TNonDiagQuerywiseTarget<TSamplesMapping>;
        using TStat = TAdditiveStatistic;
        using TMapping = TSamplesMapping;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        template <class TDataSet>
        TPairLogitPairwise(const TDataSet& dataSet,
                           TGpuAwareRandom& random,
                           const NCatboostOptions::TLossDescription& targetOptions,
                           const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor)
            : TParent(dataSet,
                      random) {
            CB_ENSURE(!objectiveDescriptor.Defined());
            Init(targetOptions);
        }

        TPairLogitPairwise(TPairLogitPairwise&& other)
            : TParent(std::move(other))
        {
        }

        using TParent::GetTarget;

        TAdditiveStatistic ComputeStats(const TConstVec& point,
                                        const TMap<TString, TString> params = TMap<TString, TString>()) const;

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
                                   const NCatboostOptions::TBootstrapConfig& config,
                                   bool secondDer,
                                   TNonDiagQuerywiseTargetDers* target) const;

        static constexpr bool IsMinOptimal() {
            return true;
        }

        ELossFunction GetScoreMetricType() const {
            return ELossFunction::PairLogit;
        }

        ELossFunction GetType() const {
            return ELossFunction::PairLogitPairwise;
        }

        EHessianType GetHessianType() const {
            return EHessianType::Symmetric;
        }

        static constexpr EOracleType OracleType() {
            return EOracleType::Pairwise;
        }

        void FillPairsAndWeightsAtPoint(const TConstVec&,
                                        TStripeBuffer<uint2>* pairs,
                                        TStripeBuffer<float>* pairWeights) const;

        void ApproximateAt(const TConstVec& point,
                           const TStripeBuffer<uint2>& pairs,
                           const TStripeBuffer<float>& pairWeights,
                           const TStripeBuffer<ui32>& scatterDerIndices,
                           TStripeBuffer<float>* value,
                           TStripeBuffer<float>* der,
                           TStripeBuffer<float>* pairDer2) const;

    private:
        void Init(const NCatboostOptions::TLossDescription& targetOptions) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::PairLogitPairwise);
        }

        double GetPairsTotalWeight() const;

    private:
        mutable double PairsTotalWeight = 0;
    };

}
