#include "pfound_f.h"
#include "kernel.h"

#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/cuda_util/bootstrap.h>
#include <catboost/cuda/gpu_data/non_zero_filter.h>

namespace NCatboostCuda {
    void TPFoundF<NCudaLib::TStripeMapping>::ApproximateStochastic(
        const TPFoundF<NCudaLib::TStripeMapping>::TConstVec& point,
        const NCatboostOptions::TBootstrapConfig& bootstrapConfig, bool,
        TNonDiagQuerywiseTargetDers* target) const {
        {
            auto& querywiseSampler = GetQueriesSampler();
            const auto& sampledGrouping = TParent::GetSamplesGrouping();
            auto& qids = querywiseSampler.GetPerDocQids(sampledGrouping);

            TCudaBuffer<uint2, TMapping> tempPairs;
            auto& weights = target->PairDer2OrWeights;

            auto& gradient = target->PointWeightedDer;
            auto& sampledDocs = target->Docs;

            target->PointDer2OrWeights.Clear();

            double queriesSampleRate = 1.0;
            double docwiseSampleRate = 1.0;
            ESamplingUnit samplingUnit = bootstrapConfig.GetSamplingUnit();

            if (bootstrapConfig.GetBootstrapType() == EBootstrapType::Bernoulli) {
                const double sampleRate = bootstrapConfig.GetTakenFraction();
                if (samplingUnit == ESamplingUnit::Group) {
                    queriesSampleRate = sampleRate;
                } else {
                    CB_ENSURE(samplingUnit == ESamplingUnit::Object);
                    docwiseSampleRate = sampleRate;
                }
            }

            if (bootstrapConfig.GetBootstrapType() == EBootstrapType::Poisson) {
                ythrow TCatBoostException() << "Poisson bootstrap is not supported for YetiRankPairwise";
            }

            {
                auto guard = NCudaLib::GetProfiler().Profile("Queries sampler in -YetiRankPairwise");
                querywiseSampler.SampleQueries(TParent::GetRandom(),
                                               queriesSampleRate,
                                               docwiseSampleRate,
                                               GetMaxQuerySize(),
                                               sampledGrouping,
                                               &sampledDocs);
            }

            TCudaBuffer<ui32, TMapping> sampledQids;
            TCudaBuffer<ui32, TMapping> sampledQidOffsets;

            ComputeQueryOffsets(qids,
                                sampledDocs,
                                &sampledQids,
                                &sampledQidOffsets);

            TCudaBuffer<ui64, TMapping> matrixOffsets;
            matrixOffsets.Reset(sampledQidOffsets.GetMapping());
            {
                auto tmp = TCudaBuffer<ui32, TMapping>::CopyMapping(matrixOffsets);
                ComputeMatrixSizes(sampledQidOffsets,
                                   &tmp);
                ScanVector(tmp, matrixOffsets);
            }

            {
                auto guard = NCudaLib::GetProfiler().Profile("Make pairs");

                tempPairs.Reset(CreateMappingFromTail(matrixOffsets, 0));
                MakePairs(sampledQidOffsets,
                          matrixOffsets,
                          &tempPairs);
            }

            weights.Reset(tempPairs.GetMapping());
            FillBuffer(weights, 0.0f);

            gradient.Reset(sampledDocs.GetMapping());
            FillBuffer(gradient, 0.0f);

            auto expApprox = TCudaBuffer<float, TMapping>::CopyMapping(sampledDocs);
            Gather(expApprox, point, sampledDocs);

            auto targets = TCudaBuffer<float, TMapping>::CopyMapping(sampledDocs);
            auto querywiseWeights = TCudaBuffer<float, TMapping>::CopyMapping(sampledDocs); //this are queryWeights

            Gather(targets, GetTarget().GetTargets(), sampledDocs);
            Gather(querywiseWeights, GetTarget().GetWeights(), sampledDocs);

            RemoveQueryMax(sampledQids,
                           sampledQidOffsets,
                           &expApprox);
            ExpVector(expApprox);

            {
                auto guard = NCudaLib::GetProfiler().Profile("PFoundFWeights");
                ComputePFoundFWeightsMatrix(DistributedSeed(TParent::GetRandom()),
                                            GetDecay(),
                                            GetPFoundPermutationCount(),
                                            expApprox,
                                            targets,
                                            sampledQids,
                                            sampledQidOffsets,
                                            matrixOffsets,
                                            &weights);

                if (bootstrapConfig.GetBootstrapType() == EBootstrapType::Bayesian) {
                    auto& seeds = TParent::GetRandom().template GetGpuSeeds<NCudaLib::TStripeMapping>();
                    BayesianBootstrap(seeds,
                                      weights,
                                      bootstrapConfig.GetBaggingTemperature());
                }
            }

            TCudaBuffer<ui64, TMapping> nzWeightPairIndices;

            FilterZeroEntries(&weights,
                              &nzWeightPairIndices);

            auto& pairs = target->Pairs;
            pairs.Reset(weights.GetMapping());
            Gather(pairs, tempPairs, nzWeightPairIndices);

            {
                auto guard = NCudaLib::GetProfiler().Profile("PFoundFMakePairsAndPointwiseGradient");

                MakeFinalPFoundGradients(sampledDocs,
                                         expApprox,
                                         querywiseWeights,
                                         targets,
                                         &weights,
                                         &pairs,
                                         &gradient);
            }
        }

        //TODO(noxoomo): maybe force defragmentation
        //TODO(noxoomo): check gradients filtering profits
    }

    void TPFoundF<NCudaLib::TStripeMapping>::FillPairsAndWeightsAtPoint(
        const TPFoundF<NCudaLib::TStripeMapping>::TConstVec& point, TStripeBuffer<uint2>* pairs,
        TStripeBuffer<float>* pairWeights) const {
        //TODO(noxoomo): here we have some overhead for final pointwise gradient computations that won't be used
        NCatboostOptions::TBootstrapConfig bootstrapConfig(ETaskType::GPU);
        TNonDiagQuerywiseTargetDers nonDiagDer2;
        TStripeBuffer<float>::Swap(*pairWeights, nonDiagDer2.PairDer2OrWeights);
        TStripeBuffer<uint2>::Swap(*pairs, nonDiagDer2.Pairs);

        StochasticGradient(point,
                           bootstrapConfig,
                           &nonDiagDer2);

        TStripeBuffer<float>::Swap(*pairWeights, nonDiagDer2.PairDer2OrWeights);
        TStripeBuffer<uint2>::Swap(*pairs, nonDiagDer2.Pairs);
        SwapWrongOrderPairs(GetTarget().GetTargets(), pairs);
    }

    void TPFoundF<NCudaLib::TStripeMapping>::ApproximateAt(const TPFoundF<NCudaLib::TStripeMapping>::TConstVec& point,
                                                           const TStripeBuffer<uint2>& pairs,
                                                           const TStripeBuffer<float>& pairWeights,
                                                           const TStripeBuffer<ui32>& scatterDerIndices,
                                                           TStripeBuffer<float>* value, TStripeBuffer<float>* der,
                                                           TStripeBuffer<float>* pairDer2) const {
        PairLogitPairwise(point,
                          pairs,
                          pairWeights,
                          scatterDerIndices,
                          value,
                          der,
                          pairDer2);
    }
}
