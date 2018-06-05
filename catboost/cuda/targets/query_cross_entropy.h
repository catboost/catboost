#pragma once

#include "query_cross_entropy_kernels.h"
#include "target_func.h"
#include "kernel.h"
#include "non_diag_target_der.h"
#include "non_diagonal_oralce_type.h"

#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/cuda/gpu_data/dataset_base.h>
#include <catboost/cuda/gpu_data/querywise_helper.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/methods/helpers.h>
#include <catboost/libs/options/bootstrap_options.h>

namespace NCatboostCuda {
    template <class TMapping, class TDataSet>
    class TQueryCrossEntropy;

    template <class TDataSet>
    class TQueryCrossEntropy<NCudaLib::TStripeMapping, TDataSet>
       : public TNonDiagQuerywiseTarget<NCudaLib::TStripeMapping, TDataSet> {
    public:
        using TSamplesMapping = NCudaLib::TStripeMapping;
        using TParent = TNonDiagQuerywiseTarget<TSamplesMapping, TDataSet>;
        using TStat = TAdditiveStatistic;
        using TMapping = TSamplesMapping;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TQueryCrossEntropy(const TDataSet& dataSet,
                           TGpuAwareRandom& random,
                           const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random) {
            Init(targetOptions);
        }

        TQueryCrossEntropy(TQueryCrossEntropy&& other)
            : TParent(std::move(other))
            , Alpha(other.Alpha)
        {
        }

        using TParent::GetTarget;

        TAdditiveStatistic ComputeStats(const TConstVec& point, double alpha) const {
            const auto& cachedData = GetCachedMetadata();

            auto funcValue = TStripeBuffer<float>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(1));
            auto orderdPoint = TStripeBuffer<float>::CopyMapping(point);
            Gather(orderdPoint, point, cachedData.FuncValueOrder);

            QueryCrossEntropy<TMapping>(alpha,
                                        cachedData.FuncValueTarget,
                                        cachedData.FuncValueWeights,
                                        orderdPoint,
                                        cachedData.FuncValueQids,
                                        cachedData.FuncValueFlags,
                                        cachedData.FuncValueQidOffsets,
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
            return MakeSimpleAdditiveStatistic(value, TParent::GetTotalWeight());
        }

        TAdditiveStatistic ComputeStats(const TConstVec& point) const {
            return ComputeStats(point, Alpha);
        }

        TAdditiveStatistic ComputeStats(const TConstVec& point,
                                        const TMap<TString, TString>& params) const {
            return ComputeStats(point, NCatboostOptions::GetAlphaQueryCrossEntropy(params));
        }

        static double Score(const TAdditiveStatistic& score) {
            return -score.Stats[0] / score.Stats[1];
        }

        double Score(const TConstVec& point) const {
            return Score(ComputeStats(point));
        }

        void StochasticGradient(const TConstVec&,
                                const NCatboostOptions::TBootstrapConfig&,
                                TNonDiagQuerywiseTargetDers*) const {
            CB_ENSURE(false, "Stochastic gradient is useless for LLMax");
        }

        void StochasticNewton(const TConstVec& point,
                              const NCatboostOptions::TBootstrapConfig& config,
                              TNonDiagQuerywiseTargetDers* target) const {
            ApproximateStochastic(point, config, target);
        }

        void ApproximateStochastic(const TConstVec& point,
                                   const NCatboostOptions::TBootstrapConfig& bootstrapConfig,
                                   TNonDiagQuerywiseTargetDers* target) const {
            auto& querywiseSampler = GetQueriesSampler();
            const auto meanQuerySize = GetMeanQuerySize();
            const auto& samplesGrouping = TParent::GetSamplesGrouping();

            auto& sampledDocs = target->Docs;
            auto& pairDer2 = target->PairDer2OrWeights;
            auto& pairs = target->Pairs;

            double queriesSampleRate = 1.0;
            if (bootstrapConfig.GetBootstrapType() == EBootstrapType::Bernoulli) {
                queriesSampleRate = bootstrapConfig.GetTakenFraction();
            }

            if (bootstrapConfig.GetBootstrapType() == EBootstrapType::Poisson) {
                ythrow TCatboostException() << "Poisson bootstrap is not supported for LLMax";
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

                MakeQidsForLLMax(&sampledDocs,
                                 &sampledQids,
                                 &sampledQidOffsets,
                                 &sampledFlags);

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
                                                sampledTarget,
                                                sampledWeights,
                                                sampledPoint,
                                                sampledQids,
                                                sampledFlags,
                                                sampledQidOffsets,
                                                nullptr,
                                                &sampledGradient,
                                                &sampledDer2,
                                                &shiftedDer2,
                                                &groupDer2);
                }

                auto matrixOffsets = TCudaBuffer<ui32, TMapping>::CopyMapping(sampledQidOffsets);

                {
                    auto tmp = TCudaBuffer<ui32, TMapping>::CopyMapping(matrixOffsets);
                    ComputeQueryLogitMatrixSizes(sampledQidOffsets,
                                                 sampledFlags,
                                                 &tmp);

                    ScanVector(tmp,
                               matrixOffsets);
                }

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

        void CreateSecondDerMatrix(NCudaLib::TCudaBuffer<uint2, NCudaLib::TStripeMapping>* pairs) const {
            const auto& cachedData = GetCachedMetadata();

            auto matrixOffsets = TCudaBuffer<ui32, TMapping>::CopyMapping(cachedData.FuncValueQidOffsets);

            {
                auto tmp = TCudaBuffer<ui32, TMapping>::CopyMapping(matrixOffsets);
                ComputeQueryLogitMatrixSizes(cachedData.FuncValueQidOffsets,
                                             cachedData.FuncValueFlags,
                                             &tmp);

                ScanVector(tmp,
                           matrixOffsets);
            }

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

        TStripeBuffer<const ui32> GetApproximateQids() const {
            const auto& cachedData = GetCachedMetadata();
            return cachedData.FuncValueQids.ConstCopyView();
        }

        TStripeBuffer<const float> GetApproximateOrderWeights() const {
            const auto& cachedData = GetCachedMetadata();
            return cachedData.FuncValueWeights.ConstCopyView();
        }

        TStripeBuffer<const ui32> GetApproximateQidOffsets() const {
            const auto& cachedData = GetCachedMetadata();
            return cachedData.FuncValueQidOffsets.ConstCopyView();
        }

        TStripeBuffer<const ui32> GetApproximateDocOrder() const {
            const auto& cachedData = GetCachedMetadata();
            return cachedData.FuncValueOrder;
        }

        void ApproximateAt(const TConstVec& orderedPoint,
                           TStripeBuffer<float>* score,
                           TStripeBuffer<float>* der,
                           TStripeBuffer<float>* pointDer2,
                           TStripeBuffer<float>* groupDer2,
                           TStripeBuffer<float>* groupSumDer2) const {
            const auto& cachedData = GetCachedMetadata();

            QueryCrossEntropy<TMapping>(Alpha,
                                        cachedData.FuncValueTarget,
                                        cachedData.FuncValueWeights,
                                        orderedPoint,
                                        cachedData.FuncValueQids,
                                        cachedData.FuncValueFlags,
                                        cachedData.FuncValueQidOffsets,
                                        score,
                                        der,
                                        pointDer2,
                                        groupDer2,
                                        groupSumDer2);
        }

        static constexpr bool IsMinOptimal() {
            return true;
        }

        TStringBuf ScoreMetricName() {
            return TStringBuilder() << "QueryCrossEntropy:alpha=" << Alpha;
        }

        ELossFunction GetScoreMetricType() const {
            return ELossFunction::QueryCrossEntropy;
        }

        static constexpr ENonDiagonalOracleType NonDiagonalOracleType() {
            return ENonDiagonalOracleType::Groupwise;
        }


    private:
        struct TQueryLogitApproxHelpData {
            TCudaBuffer<float, TMapping> FuncValueTarget;
            TCudaBuffer<float, TMapping> FuncValueWeights;
            TCudaBuffer<ui32, TMapping> FuncValueOrder;
            TCudaBuffer<bool, TMapping> FuncValueFlags;
            TCudaBuffer<ui32, TMapping> FuncValueQids;
            TCudaBuffer<ui32, TMapping> FuncValueQidOffsets;
        };

    private:
        ui32 GetMaxQuerySize() const {
            return 256;
        }

        bool HasBigQueries() const {
            return false;
        }

        void Init(const NCatboostOptions::TLossDescription& targetOptions) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::QueryCrossEntropy);
            Alpha = NCatboostOptions::GetAlphaQueryCrossEntropy(targetOptions);
        }

        TQuerywiseSampler& GetQueriesSampler() const {
            if (QueriesSampler == nullptr) {
                QueriesSampler = new TQuerywiseSampler();
            }
            return *QueriesSampler;
        }

        void MakeQidsForLLMax(TStripeBuffer<ui32>* order,
                              TStripeBuffer<ui32>* orderQids,
                              TStripeBuffer<ui32>* orderQidOffsets,
                              TStripeBuffer<bool>* flags) const {
            const auto& samplesGrouping = TParent::GetSamplesGrouping();
            double meanQuerySize = GetMeanQuerySize();
            const auto& qids = GetQueriesSampler().GetPerDocQids(samplesGrouping);

            ComputeQueryOffsets(qids,
                                *order,
                                orderQids,
                                orderQidOffsets);

            flags->Reset(order->GetMapping());

            MakeIsSingleClassQueryFlags(GetTarget().GetTargets(),
                                        order->ConstCopyView(),
                                        orderQidOffsets->ConstCopyView(),
                                        meanQuerySize,
                                        flags);

            RadixSort(*flags,
                      *order,
                      false,
                      0,
                      1);

            ComputeQueryOffsets(qids,
                                *order,
                                orderQids,
                                orderQidOffsets);
        }

        const TQueryLogitApproxHelpData& GetCachedMetadata() const {
            if (CachedMetadata.FuncValueOrder.GetObjectsSlice().Size() == 0) {
                CachedMetadata.FuncValueOrder.Reset(GetTarget().GetTargets().GetMapping());
                GetTarget().WriteIndices(CachedMetadata.FuncValueOrder);

                MakeQidsForLLMax(&CachedMetadata.FuncValueOrder,
                                 &CachedMetadata.FuncValueQids,
                                 &CachedMetadata.FuncValueQidOffsets,
                                 &CachedMetadata.FuncValueFlags);
                CachedMetadata.FuncValueTarget = TStripeBuffer<float>::CopyMapping(CachedMetadata.FuncValueOrder);
                CachedMetadata.FuncValueWeights = TStripeBuffer<float>::CopyMapping(CachedMetadata.FuncValueOrder);
                Gather(CachedMetadata.FuncValueTarget, GetTarget().GetTargets(), CachedMetadata.FuncValueOrder);
                Gather(CachedMetadata.FuncValueWeights, GetTarget().GetWeights(), CachedMetadata.FuncValueOrder);
            }
            return CachedMetadata;
        }

        double GetMeanQuerySize() const {
            double totalDocs = GetTarget().GetTargets().GetObjectsSlice().Size();
            double totalQueries = TParent::GetSamplesGrouping().GetQueryCount();
            return totalQueries > 0 ? totalDocs / totalQueries : 0;
        }

    private:
        mutable THolder<TQuerywiseSampler> QueriesSampler;
        double Alpha;
        mutable TQueryLogitApproxHelpData CachedMetadata;
    };

}
