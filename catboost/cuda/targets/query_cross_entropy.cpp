#include "query_cross_entropy.h"
#include "query_cross_entropy_kernels.h"

#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/cuda_util/bootstrap.h>
#include <catboost/cuda/cuda_util/helpers.h>

namespace NCatboostCuda {
    TAdditiveStatistic TQueryCrossEntropy<NCudaLib::TStripeMapping>::ComputeStats(
        const TQueryCrossEntropy<NCudaLib::TStripeMapping>::TConstVec& point, double alpha) const {
        const auto& cachedData = GetCachedMetadata();

        auto funcValue = TStripeBuffer<float>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(1));
        auto orderdPoint = TStripeBuffer<float>::CopyMapping(point);
        Gather(orderdPoint, point, cachedData.FuncValueOrder);

        QueryCrossEntropy<TMapping>(alpha,
                                    DefaultScale,
                                    ApproxScaleSize,
                                    cachedData.FuncValueTarget.AsConstBuf(),
                                    cachedData.FuncValueWeights.AsConstBuf(),
                                    orderdPoint.AsConstBuf(),
                                    cachedData.FuncValueQids,
                                    cachedData.FuncValueFlags,
                                    cachedData.FuncValueQidOffsets,
                                    ApproxScale.AsConstBuf(),
                                    cachedData.TrueClassCount,
                                    &funcValue,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr);

        TVector<float> resultCpu;
        funcValue.Read(resultCpu);

        double value = 0;
        for (auto deviceVal : resultCpu) {
            value += deviceVal;
        }
        return MakeSimpleAdditiveStatistic(-value, TParent::GetTotalWeight());
    }

    const TQueryCrossEntropy<NCudaLib::TStripeMapping>::TQueryLogitApproxHelpData&
    TQueryCrossEntropy<NCudaLib::TStripeMapping>::GetCachedMetadata() const {
        if (CachedMetadata.FuncValueOrder.GetObjectsSlice().Size() == 0) {
            CachedMetadata.FuncValueOrder.Reset(GetTarget().GetTargets().GetMapping());
            GetTarget().WriteIndices(CachedMetadata.FuncValueOrder);

            MakeQidsForLLMax(&CachedMetadata.FuncValueOrder,
                             &CachedMetadata.FuncValueQids,
                             &CachedMetadata.FuncValueQidOffsets,
                             &CachedMetadata.FuncValueFlags,
                             &CachedMetadata.TrueClassCount);
            CachedMetadata.FuncValueTarget = TStripeBuffer<float>::CopyMapping(CachedMetadata.FuncValueOrder);
            CachedMetadata.FuncValueWeights = TStripeBuffer<float>::CopyMapping(CachedMetadata.FuncValueOrder);
            Gather(CachedMetadata.FuncValueTarget, GetTarget().GetTargets(), CachedMetadata.FuncValueOrder);
            Gather(CachedMetadata.FuncValueWeights, GetTarget().GetWeights(), CachedMetadata.FuncValueOrder);
        }
        return CachedMetadata;
    }

    void TQueryCrossEntropy<NCudaLib::TStripeMapping>::MakeQidsForLLMax(TStripeBuffer<ui32>* order,
                                                                        TStripeBuffer<ui32>* orderQids,
                                                                        TStripeBuffer<ui32>* orderQidOffsets,
                                                                        TStripeBuffer<bool>* flags,
                                                                        TStripeBuffer<ui32>* trueClassCount) const {
        const auto& samplesGrouping = TParent::GetSamplesGrouping();
        double meanQuerySize = GetMeanQuerySize();
        const auto& qids = GetQueriesSampler().GetPerDocQids(samplesGrouping);

        ComputeQueryOffsets(qids,
                            *order,
                            orderQids,
                            orderQidOffsets);

        flags->Reset(order->GetMapping());
        trueClassCount->Reset(order->GetMapping());

        MakeIsSingleClassQueryFlags(GetTarget().GetTargets(),
                                    order->ConstCopyView(),
                                    orderQidOffsets->ConstCopyView(),
                                    meanQuerySize,
                                    flags,
                                    trueClassCount);
        auto unorderedFlags = TStripeBuffer<bool>::CopyMapping(*flags);
        unorderedFlags.Copy(*flags);

        RadixSort(*flags,
                  *order,
                  false,
                  0,
                  1);

        RadixSort(unorderedFlags,
                  *trueClassCount,
                  false,
                  0,
                  1);

        ComputeQueryOffsets(qids,
                            *order,
                            orderQids,
                            orderQidOffsets);
    }

    void TQueryCrossEntropy<NCudaLib::TStripeMapping>::ApproximateAt(const TQueryCrossEntropy<NCudaLib::TStripeMapping>::TConstVec& orderedPoint,
                                                                     TStripeBuffer<float>* score,
                                                                     TStripeBuffer<float>* der,
                                                                     TStripeBuffer<float>* pointDer2,
                                                                     TStripeBuffer<float>* groupDer2,
                                                                     TStripeBuffer<float>* groupSumDer2) const {
        const auto& cachedData = GetCachedMetadata();

        QueryCrossEntropy<TMapping>(Alpha,
                                    DefaultScale,
                                    ApproxScaleSize,
                                    cachedData.FuncValueTarget.AsConstBuf(),
                                    cachedData.FuncValueWeights.AsConstBuf(),
                                    orderedPoint,
                                    cachedData.FuncValueQids,
                                    cachedData.FuncValueFlags,
                                    cachedData.FuncValueQidOffsets,
                                    ApproxScale.AsConstBuf(),
                                    cachedData.TrueClassCount,
                                    score,
                                    der,
                                    pointDer2,
                                    groupDer2,
                                    groupSumDer2);
    }

    template <typename TMapping>
    static TCudaBuffer<ui64, TMapping> CalcMatrixOffsets(
        const TCudaBuffer<ui32, TMapping>& queryOffsets,
        const TCudaBuffer<bool, TMapping>& queryFlags
    ) {
        auto matrixSizes = TCudaBuffer<ui32, TMapping>::CopyMapping(queryOffsets);
        ComputeQueryLogitMatrixSizes(queryOffsets, queryFlags, &matrixSizes);
        auto matrixOffsets = TCudaBuffer<ui64, TMapping>::CopyMapping(queryOffsets);
        ScanVector(matrixSizes, matrixOffsets);

        return matrixOffsets;
    }

    void TQueryCrossEntropy<NCudaLib::TStripeMapping>::CreateSecondDerMatrix(
        NCudaLib::TCudaBuffer<uint2, NCudaLib::TStripeMapping>* pairs) const {
        const auto& cachedData = GetCachedMetadata();

        auto matrixOffsets = CalcMatrixOffsets(cachedData.FuncValueQidOffsets, cachedData.FuncValueFlags);

        {
            auto guard = NCudaLib::GetProfiler().Profile("Make pairs");

            pairs->Reset(CreateMappingFromTail(matrixOffsets, 0));
            MakePairsQueryLogit<NCudaLib::TStripeMapping>(cachedData.FuncValueQidOffsets,
                                                          matrixOffsets,
                                                          cachedData.FuncValueFlags,
                                                          GetMeanQuerySize(),
                                                          pairs);
        }
    }

    void TQueryCrossEntropy<NCudaLib::TStripeMapping>::ApproximateStochastic(
        const TQueryCrossEntropy<NCudaLib::TStripeMapping>::TConstVec& point,
        const NCatboostOptions::TBootstrapConfig& bootstrapConfig, TNonDiagQuerywiseTargetDers* target) const {
        auto& querywiseSampler = GetQueriesSampler();
        const auto meanQuerySize = GetMeanQuerySize();
        const auto& samplesGrouping = TParent::GetSamplesGrouping();

        auto& sampledDocs = target->Docs;
        auto& pairDer2 = target->PairDer2OrWeights; // size can exceed 32-bits
        auto& pairs = target->Pairs; // size can exceed 32-bits

        double queriesSampleRate = 1.0;
        if (bootstrapConfig.GetBootstrapType() == EBootstrapType::Bernoulli) {
            queriesSampleRate = bootstrapConfig.GetTakenFraction();
        } else {
            CB_ENSURE(bootstrapConfig.GetBootstrapType() == EBootstrapType::No,
                bootstrapConfig.GetBootstrapType() << " bootstrap is not supported for LLMax");
        }

        if (queriesSampleRate < 1.0 || HasBigQueries()) {
            auto guard = NCudaLib::GetProfiler().Profile("Queries sampler in LLMax");
            querywiseSampler.SampleQueries(TParent::GetRandom(),
                                           queriesSampleRate,
                                           1.0,
                                           GetMaxQuerySize(),
                                           samplesGrouping,
                                           &sampledDocs);
        } else {
            sampledDocs.Reset(GetTarget().GetTargets().GetMapping());
            GetTarget().WriteIndices(sampledDocs);
        }

        auto sampledGradient = TStripeBuffer<float>::CopyMapping(sampledDocs);
        auto sampledDer2 = TStripeBuffer<float>::CopyMapping(sampledDocs);

        {
            auto shiftedDer2 = TStripeBuffer<float>::CopyMapping(sampledDocs);

            TCudaBuffer<ui32, TMapping> sampledQids;
            TCudaBuffer<ui32, TMapping> sampledQidOffsets;
            TCudaBuffer<bool, TMapping> sampledFlags;
            TCudaBuffer<ui32, TMapping> sampledTrueClassCount;

            MakeQidsForLLMax(&sampledDocs,
                             &sampledQids,
                             &sampledQidOffsets,
                             &sampledFlags,
                             &sampledTrueClassCount);

            TStripeBuffer<float> groupDer2 = TStripeBuffer<float>::CopyMapping(sampledQidOffsets);

            {
                auto sampledTarget = TStripeBuffer<float>::CopyMapping(sampledDocs);
                Gather(sampledTarget, GetTarget().GetTargets(), sampledDocs);

                auto sampledWeights = TStripeBuffer<float>::CopyMapping(sampledDocs);
                Gather(sampledWeights, GetTarget().GetWeights(), sampledDocs);

                auto sampledPoint = TStripeBuffer<float>::CopyMapping(sampledDocs);
                Gather(sampledPoint, point, sampledDocs);

                if (bootstrapConfig.GetBootstrapType() == EBootstrapType::Bayesian) {
                    auto& seeds = TParent::GetRandom().template GetGpuSeeds<NCudaLib::TStripeMapping>();
                    BayesianBootstrap(seeds,
                                      sampledWeights,
                                      bootstrapConfig.GetBaggingTemperature());
                }

                QueryCrossEntropy<TMapping>(Alpha,
                                            DefaultScale,
                                            ApproxScaleSize,
                                            sampledTarget.AsConstBuf(),
                                            sampledWeights.AsConstBuf(),
                                            sampledPoint.AsConstBuf(),
                                            sampledQids,
                                            sampledFlags,
                                            sampledQidOffsets,
                                            ApproxScale.AsConstBuf(),
                                            sampledTrueClassCount,
                                            nullptr,
                                            &sampledGradient,
                                            &sampledDer2,
                                            &shiftedDer2,
                                            &groupDer2);
            }

            auto matrixOffsets = CalcMatrixOffsets(sampledQidOffsets, sampledFlags);

            {
                auto guard = NCudaLib::GetProfiler().Profile("Make pairs");

                pairs.Reset(CreateMappingFromTail(matrixOffsets, 0));
                MakePairsQueryLogit(sampledQidOffsets,
                                    matrixOffsets,
                                    sampledFlags,
                                    meanQuerySize,
                                    &pairs);
            }

            pairDer2.Reset(pairs.GetMapping());
            FillPairDer2AndRemapPairDocuments(shiftedDer2,
                                              groupDer2,
                                              sampledDocs,
                                              sampledQids,
                                              &pairDer2,
                                              &pairs);
        }

        auto tmpIndices = TStripeBuffer<ui32>::CopyMapping(sampledDocs);
        MakeSequence(tmpIndices);
        //for faster histograms
        RadixSort(sampledDocs, tmpIndices);

        auto& gradient = target->PointWeightedDer;
        auto& pointDer2 = target->PointDer2OrWeights;

        gradient.Reset(sampledDocs.GetMapping());
        pointDer2.Reset(sampledDocs.GetMapping());

        Gather(gradient, sampledGradient, tmpIndices);
        Gather(pointDer2, sampledDer2, tmpIndices);
    }

}
