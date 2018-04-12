#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <catboost/cuda/cuda_util/kernel/transform.cuh>
#include <catboost/cuda/targets/kernel/pointwise_targets.cuh>
#include <catboost/cuda/targets/kernel/query_rmse.cuh>
#include <catboost/cuda/targets/kernel/query_softmax.cuh>
#include <catboost/cuda/targets/kernel/pair_logit.cuh>
#include <catboost/cuda/targets/kernel/yeti_rank_pointwise.cuh>

namespace NKernelHost {
    class TCrossEntropyTargetKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> TargetClasses;
        TCudaBufferPtr<const float> TargetWeights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;
        TCudaBufferPtr<float> Der2;
        float Border;
        bool UseBorder;

    public:
        TCrossEntropyTargetKernel() = default;

        TCrossEntropyTargetKernel(TCudaBufferPtr<const float> targetClasses, TCudaBufferPtr<const float> targetWeights, TCudaBufferPtr<const float> predictions, TCudaBufferPtr<float> functionValue, TCudaBufferPtr<float> der, TCudaBufferPtr<float> der2, float border, bool useBorder)
            : TargetClasses(targetClasses)
            , TargetWeights(targetWeights)
            , Predictions(predictions)
            , FunctionValue(functionValue)
            , Der(der)
            , Der2(der2)
            , Border(border)
            , UseBorder(useBorder)
        {
        }

        Y_SAVELOAD_DEFINE(TargetClasses, TargetWeights, Predictions, FunctionValue, Der, Der2, Border, UseBorder);

        void Run(const TCudaStream& stream) const {
            NKernel::CrossEntropyTargetKernel(TargetClasses.Get(), TargetWeights.Get(), TargetClasses.Size(), Predictions.Get(), FunctionValue.Get(), Der.Get(), Der2.Get(), Border, UseBorder, stream.GetStream());
        }
    };

    class TMseTargetKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Relevs;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;
        TCudaBufferPtr<float> Der2;

    public:
        TMseTargetKernel() = default;

        TMseTargetKernel(TCudaBufferPtr<const float> relevs, TCudaBufferPtr<const float> weights, TCudaBufferPtr<const float> predictions, TCudaBufferPtr<float> functionValue, TCudaBufferPtr<float> der, TCudaBufferPtr<float> der2)
            : Relevs(relevs)
            , Weights(weights)
            , Predictions(predictions)
            , FunctionValue(functionValue)
            , Der(der)
            , Der2(der2)
        {
        }

        Y_SAVELOAD_DEFINE(Relevs, Weights, Predictions, FunctionValue, Der, Der2);

        void Run(const TCudaStream& stream) const {
            NKernel::MseTargetKernel(Relevs.Get(), Weights.Get(), static_cast<ui32>(Relevs.Size()),
                                     Predictions.Get(),
                                     FunctionValue.Get(), Der.Get(), Der2.Get(),
                                     stream.GetStream());
        }
    };

    class TPointwiseTargetImplKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Relevs;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;
        float Alpha;
        ELossFunction LossFunction;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;
        TCudaBufferPtr<float> Der2;

    public:
        TPointwiseTargetImplKernel() = default;

        TPointwiseTargetImplKernel(TCudaBufferPtr<const float> relevs,
                                   TCudaBufferPtr<const float> weights,
                                   TCudaBufferPtr<const float> predictions,
                                   float alpha,
                                   ELossFunction lossFunction,
                                   TCudaBufferPtr<float> functionValue,
                                   TCudaBufferPtr<float> der,
                                   TCudaBufferPtr<float> der2)
            : Relevs(relevs)
            , Weights(weights)
            , Predictions(predictions)
            , Alpha(alpha)
            , LossFunction(lossFunction)
            , FunctionValue(functionValue)
            , Der(der)
            , Der2(der2)
        {
        }

        Y_SAVELOAD_DEFINE(Relevs, Weights, Predictions, FunctionValue, Der, Der2, Alpha, LossFunction);

        void Run(const TCudaStream& stream) const {
            NKernel::PointwiseTargetKernel(Relevs.Get(), Weights.Get(), static_cast<ui32>(Relevs.Size()),
                                           LossFunction, Alpha,
                                           Predictions.Get(),
                                           FunctionValue.Get(), Der.Get(), Der2.Get(),
                                           stream.GetStream());
        }
    };

    class TQueryRmseKernel: public TKernelBase<NKernel::TQueryRmseContext, false> {
    private:
        TCudaBufferPtr<const ui32> QuerySizes;
        TCudaBufferPtr<const ui32> QueryOffsets;
        ui32 QueryOffsetsBias;

        TCudaBufferPtr<const float> Relevs;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;

        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;
        TCudaBufferPtr<float> Der2;

    public:
        using TKernelContext = NKernel::TQueryRmseContext;
        Y_SAVELOAD_DEFINE(Relevs, Weights, Predictions,
                          QueryOffsets, QuerySizes, QueryOffsetsBias,
                          Indices, FunctionValue,
                          Der, Der2);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            //TODO(noxoomo): make temp memory more robust
            auto queryMeansPtr = memoryManager.Allocate<float>(QuerySizes.Size());
            auto qidsPtr = memoryManager.Allocate<ui32>(Relevs.Size());
            auto relevsPtr = memoryManager.Allocate<float>(Relevs.Size());

            context->QueryMeans = queryMeansPtr.Get();
            context->Qids = qidsPtr.Get();
            context->MseDer = relevsPtr.Get();
            return context;
        }

        TQueryRmseKernel() = default;

        TQueryRmseKernel(TCudaBufferPtr<const ui32> querySizes,
                         TCudaBufferPtr<const ui32> queryOffsets,
                         ui32 queryOffsetsBias,
                         TCudaBufferPtr<const float> relevs,
                         TCudaBufferPtr<const float> weights,
                         TCudaBufferPtr<const float> predictions,
                         TCudaBufferPtr<const ui32> indices,
                         TCudaBufferPtr<float> functionValue,
                         TCudaBufferPtr<float> der,
                         TCudaBufferPtr<float> der2)
            : QuerySizes(querySizes)
            , QueryOffsets(queryOffsets)
            , QueryOffsetsBias(queryOffsetsBias)
            , Relevs(relevs)
            , Weights(weights)
            , Predictions(predictions)
            , Indices(indices)
            , FunctionValue(functionValue)
            , Der(der)
            , Der2(der2)
        {
        }

        void Run(const TCudaStream& stream, TKernelContext& context) {
            if (Der.Size()) {
                CB_ENSURE(Der.Size() == Predictions.Size());
            }
            CB_ENSURE(QuerySizes.Size() == QueryOffsets.Size());
            if (Indices.Size()) {
                CB_ENSURE(Indices.Size() == Predictions.Size());
                NKernel::Gather(context.MseDer, Predictions.Get(), Indices.Get(), Indices.Size(), stream.GetStream());
            } else {
                CopyMemoryAsync(Predictions.Get(), context.MseDer, Predictions.Size(), stream);
            }

            NKernel::MultiplyVector(context.MseDer, -1.0f, Predictions.Size(), stream.GetStream());
            NKernel::AddVector(context.MseDer, Relevs.Get(), Relevs.Size(), stream.GetStream());
            NKernel::ComputeGroupMeans(context.MseDer, Weights.Get(), QueryOffsets.Get(), QueryOffsetsBias, QuerySizes.Get(), QueryOffsets.Size(), context.QueryMeans, stream.GetStream());
            NKernel::ComputeGroupIds(QuerySizes.Get(), QueryOffsets.Get(), QueryOffsetsBias, QueryOffsets.Size(), context.Qids, stream.GetStream());
            NKernel::ApproximateQueryRmse(context.MseDer,
                                          Weights.Get(),
                                          context.Qids,
                                          static_cast<ui32>(Predictions.Size()),
                                          context.QueryMeans,
                                          Indices.Get(),
                                          FunctionValue.Get(),
                                          Der.Get(),
                                          Der2.Get(),
                                          stream.GetStream());
        }
    };

    class TQuerySoftMaxKernel: public TKernelBase<NKernel::TQuerySoftMaxContext, false> {
    private:
        TCudaBufferPtr<const ui32> QuerySizes;
        TCudaBufferPtr<const ui32> QueryOffsets;
        ui32 QueryOffsetsBias;

        TCudaBufferPtr<const float> Relevs;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;

        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;
        TCudaBufferPtr<float> Der2;

    public:
        using TKernelContext = NKernel::TQuerySoftMaxContext;
        Y_SAVELOAD_DEFINE(Relevs, Weights, Predictions,
                          QueryOffsets, QuerySizes, QueryOffsetsBias,
                          Indices, FunctionValue,
                          Der, Der2);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            auto approxExpPtr = memoryManager.Allocate<float>(Relevs.Size());
            auto queryApproxPtr = memoryManager.Allocate<float>(QuerySizes.Size());
            auto querySumWeightedTargetsPtr = memoryManager.Allocate<float>(QuerySizes.Size());
            auto qidsPtr = memoryManager.Allocate<ui32>(Relevs.Size());

            context->ApproxExp = approxExpPtr.Get();
            context->QueryApprox = queryApproxPtr.Get();
            context->QuerySumWeightedTargets = querySumWeightedTargetsPtr.Get();
            context->Qids = qidsPtr.Get();
            return context;
        }

        TQuerySoftMaxKernel() = default;

        TQuerySoftMaxKernel(TCudaBufferPtr<const ui32> querySizes,
                            TCudaBufferPtr<const ui32> queryOffsets,
                            ui32 queryOffsetsBias,
                            TCudaBufferPtr<const float> relevs,
                            TCudaBufferPtr<const float> weights,
                            TCudaBufferPtr<const float> predictions,
                            TCudaBufferPtr<const ui32> indices,
                            TCudaBufferPtr<float> functionValue,
                            TCudaBufferPtr<float> der,
                            TCudaBufferPtr<float> der2)
            : QuerySizes(querySizes)
            , QueryOffsets(queryOffsets)
            , QueryOffsetsBias(queryOffsetsBias)
            , Relevs(relevs)
            , Weights(weights)
            , Predictions(predictions)
            , Indices(indices)
            , FunctionValue(functionValue)
            , Der(der)
            , Der2(der2)
        {
        }

        void Run(const TCudaStream& stream, TKernelContext& context) {
            if (Der.Size()) {
                CB_ENSURE(Der.Size() == Predictions.Size());
            }
            CB_ENSURE(QuerySizes.Size() == QueryOffsets.Size());
            if (Indices.Size()) {
                CB_ENSURE(Indices.Size() == Predictions.Size());
                NKernel::Gather(context.ApproxExp, Predictions.Get(), Indices.Get(), Indices.Size(), stream.GetStream());
            } else {
                CopyMemoryAsync(Predictions.Get(), context.ApproxExp, Predictions.Size(), stream);
            }

            NKernel::ComputeGroupIds(QuerySizes.Get(), QueryOffsets.Get(), QueryOffsetsBias, QueryOffsets.Size(), context.Qids, stream.GetStream());
            NKernel::ComputeGroupMaximals(Relevs.Get(),
                                          Weights.Get(),
                                          context.ApproxExp,
                                          QueryOffsets.Get(),
                                          QueryOffsetsBias,
                                          QuerySizes.Get(),
                                          QueryOffsets.Size(),
                                          context.QueryApprox,
                                          context.QuerySumWeightedTargets,
                                          stream.GetStream());
            NKernel::ComputeQueryExponents(Weights.Get(),
                                           context.Qids,
                                           static_cast<ui32>(Predictions.Size()),
                                           context.QueryApprox,
                                           Indices.Get(),
                                           context.ApproxExp,
                                           stream.GetStream());
            NKernel::ComputeGroupSums(context.ApproxExp,
                                      QueryOffsets.Get(),
                                      QueryOffsetsBias,
                                      QuerySizes.Get(),
                                      QueryOffsets.Size(),
                                      context.QueryApprox,
                                      stream.GetStream());
            NKernel::ApproximateQuerySoftMax(Relevs.Get(),
                                             Weights.Get(),
                                             context.ApproxExp,
                                             context.Qids,
                                             static_cast<ui32>(Predictions.Size()),
                                             context.QueryApprox,
                                             context.QuerySumWeightedTargets,
                                             Indices.Get(),
                                             FunctionValue.Get(),
                                             Der.Get(),
                                             Der2.Get(),
                                             stream.GetStream());
        }
    };

    class TYetiRankKernel: public TKernelBase<NKernel::TYetiRankContext, false> {
    private:
        TCudaBufferPtr<const ui32> QuerySizes;
        TCudaBufferPtr<const ui32> QueryOffsets;
        ui32 QueryOffsetsBias;

        TCudaBufferPtr<const float> Relevs;
        TCudaBufferPtr<const float> Predictions;

        ui64 Seed;
        ui32 PermutationCount;

        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;
        TCudaBufferPtr<float> Der2;

    public:
        using TKernelContext = NKernel::TYetiRankContext;
        Y_SAVELOAD_DEFINE(QueryOffsets, QuerySizes, QueryOffsetsBias,
                          Relevs, Predictions,
                          Seed, PermutationCount,
                          Indices, FunctionValue, Der, Der2);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            //TODO(noxoomo): make temp memory more robust
            auto context = MakeHolder<TKernelContext>();
            auto queryMeans = memoryManager.Allocate<float>(QuerySizes.Size());
            auto qids = memoryManager.Allocate<ui32>(Relevs.Size());
            auto lastProceededQid = memoryManager.Allocate<ui32>(1);
            auto approxesTemp = memoryManager.Allocate<float>(Relevs.Size());

            NCudaLib::THandleBasedMemoryPointer<float, NCudaLib::EPtrType::CudaDevice> tempDers;
            NCudaLib::THandleBasedMemoryPointer<float, NCudaLib::EPtrType::CudaDevice> tempWeights;
            if (Indices.Size()) {
                tempDers = memoryManager.Allocate<float>(Relevs.Size());
                tempWeights = memoryManager.Allocate<float>(Relevs.Size());
            }

            //now ptrs would not change until kernel is finished
            context->QueryMeans = queryMeans.Get();
            context->Qids = qids.Get();
            context->LastProceededQid = lastProceededQid.Get();
            context->Approxes = approxesTemp.Get();

            if (Indices.Size()) {
                context->TempDers = tempDers.Get();
                context->TempWeights = tempWeights.Get();
            }
            return context;
        }

        TYetiRankKernel() = default;

        TYetiRankKernel(TCudaBufferPtr<const ui32> querySizes,
                        TCudaBufferPtr<const ui32> queryOffsets,
                        ui32 queryOffsetsBias,
                        TCudaBufferPtr<const float> relevs,
                        TCudaBufferPtr<const float> predictions,
                        ui64 seed, ui32 permutationCount,
                        TCudaBufferPtr<const ui32> indices,
                        TCudaBufferPtr<float> functionValue,
                        TCudaBufferPtr<float> der,
                        TCudaBufferPtr<float> der2)
            : QuerySizes(querySizes)
            , QueryOffsets(queryOffsets)
            , QueryOffsetsBias(queryOffsetsBias)
            , Relevs(relevs)
            , Predictions(predictions)
            , Seed(seed)
            , PermutationCount(permutationCount)
            , Indices(indices)
            , FunctionValue(functionValue)
            , Der(der)
            , Der2(der2)
        {
        }

        void Run(const TCudaStream& stream, TKernelContext& context) {
            CB_ENSURE(Der.Size() == Predictions.Size());
            CB_ENSURE(Der2.Size() == Predictions.Size());

            float* derDst;
            float* weightsDst;

            CB_ENSURE(QuerySizes.Size() == QueryOffsets.Size());
            if (Indices.Size()) {
                CB_ENSURE(Indices.Size() == Predictions.Size());
                NKernel::Gather(context.Approxes, Predictions.Get(), Indices.Get(), Indices.Size(), stream.GetStream());
                derDst = context.TempDers;
                weightsDst = context.TempWeights;
            } else {
                CopyMemoryAsync(Predictions.Get(), context.Approxes, Predictions.Size(), stream);
                derDst = Der.Get();
                weightsDst = Der2.Get();
            }

            //we adjust target by group means to avoid exponents with of relatively big numbers
            NKernel::ComputeGroupMeans(context.Approxes, nullptr, QueryOffsets.Get(), QueryOffsetsBias, QuerySizes.Get(), QueryOffsets.Size(), context.QueryMeans, stream.GetStream());
            NKernel::ComputeGroupIds(QuerySizes.Get(), QueryOffsets.Get(), QueryOffsetsBias, QueryOffsets.Size(), context.Qids, stream.GetStream());
            NKernel::RemoveQueryMeans((int*)(context.Qids), QuerySizes.Size(), context.QueryMeans, context.Approxes, stream.GetStream());

            if (FunctionValue.Size()) {
                NKernel::FillBuffer(FunctionValue.Get(), 0.0f, 1, stream.GetStream());
            }

            NKernel::YetiRankGradient(Seed, PermutationCount,
                                      QueryOffsets.Get(),
                                      (int*)context.LastProceededQid,
                                      QueryOffsetsBias,
                                      QueryOffsets.Size(),
                                      (int*)context.Qids,
                                      context.Approxes,
                                      Relevs.Get(),
                                      Predictions.Size(),
                                      derDst,
                                      weightsDst,
                                      stream.GetStream());

            if (Indices.Size()) {
                NKernel::Scatter(Der.Get(), context.TempDers, Indices.Get(), Der.Size(), stream.GetStream());
                NKernel::Scatter(Der2.Get(), context.TempWeights, Indices.Get(), Der.Size(), stream.GetStream());
            }
        }
    };

    class TPairLogitKernel: public TKernelBase<NKernel::TPairLogitContext, false> {
    private:
        TCudaBufferPtr<const uint2> Pairs;
        TCudaBufferPtr<const float> PairWeights;
        ui32 QueryOffsetsBias;

        TCudaBufferPtr<const float> Predictions;

        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;
        TCudaBufferPtr<float> Der2;

    public:
        using TKernelContext = NKernel::TPairLogitContext;
        Y_SAVELOAD_DEFINE(Pairs, PairWeights, Predictions, Indices, FunctionValue, Der, Der2);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            if (Indices.Get()) {
                //TODO(noxoomo): make temp memory more robust
                context->GatheredPoint = memoryManager.Allocate<float>(Indices.Size()).Get();
            }
            return context;
        }

        TPairLogitKernel() = default;

        TPairLogitKernel(TCudaBufferPtr<const uint2> pairs,
                         TCudaBufferPtr<const float> pairWeights,
                         ui32 queryOffsetsBias,
                         TCudaBufferPtr<const float> predictions,
                         TCudaBufferPtr<const ui32> indices,
                         TCudaBufferPtr<float> functionValue,
                         TCudaBufferPtr<float> der,
                         TCudaBufferPtr<float> der2)
            : Pairs(pairs)
            , PairWeights(pairWeights)
            , QueryOffsetsBias(queryOffsetsBias)
            , Predictions(predictions)
            , Indices(indices)
            , FunctionValue(functionValue)
            , Der(der)
            , Der2(der2)
        {
        }

        void Run(const TCudaStream& stream, TKernelContext& context) {
            const float* point = nullptr;
            if (Indices.Get()) {
                CB_ENSURE(Indices.Size() == Predictions.Size());
                NKernel::Gather(context.GatheredPoint, Predictions.Get(), Indices.Get(), Indices.Size(), stream.GetStream());
                point = context.GatheredPoint;
            } else {
                point = Predictions.Get();
            }

            const ui32 docCount = Predictions.Size();
            if (Der.Get()) {
                CB_ENSURE(Der.Size() == Predictions.Size());
            }
            if (Der2.Get()) {
                CB_ENSURE(Der2.Size() == Predictions.Size());
            }

            NKernel::PairLogitPointwiseTarget(point,
                                              Pairs.Get(),
                                              PairWeights.Get(),
                                              Indices.Get(),
                                              Pairs.Size(), QueryOffsetsBias,
                                              FunctionValue.Get(),
                                              Der.Get(),
                                              Der2.Get(),
                                              docCount,
                                              stream.GetStream());
        }
    };

    class TMakePairWeightsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const uint2> Pairs;
        TCudaBufferPtr<const float> PairWeights;
        TCudaBufferPtr<float> Weights;

    public:
        Y_SAVELOAD_DEFINE(Pairs, PairWeights, Weights);

        TMakePairWeightsKernel() = default;

        TMakePairWeightsKernel(TCudaBufferPtr<const uint2> pairs,
                               TCudaBufferPtr<const float> pairWeights,
                               TCudaBufferPtr<float> weights)
            : Pairs(pairs)
            , PairWeights(pairWeights)
            , Weights(weights)
        {
        }

        void Run(const TCudaStream& stream) {
            NKernel::MakePairWeights(Pairs.Get(),
                                     PairWeights.Get(),
                                     PairWeights.Size(),
                                     Weights.Get(),
                                     stream.GetStream());
        }
    };
}

template <class TMapping>
inline void ApproximateMse(const TCudaBuffer<const float, TMapping>& target,
                           const TCudaBuffer<const float, TMapping>& weights,
                           const TCudaBuffer<const float, TMapping>& point,
                           TCudaBuffer<float, TMapping>* score,
                           TCudaBuffer<float, TMapping>* weightedDer,
                           TCudaBuffer<float, TMapping>* weightedDer2,
                           ui32 stream = 0) {
    using TKernel = NKernelHost::TMseTargetKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(), stream, target, weights, point, score, weightedDer, weightedDer2);
}

template <class TMapping>
inline void ApproximatePointwise(const TCudaBuffer<const float, TMapping>& target,
                                 const TCudaBuffer<const float, TMapping>& weights,
                                 const TCudaBuffer<const float, TMapping>& point,
                                 ELossFunction lossFunction, float alpha,
                                 TCudaBuffer<float, TMapping>* score,
                                 TCudaBuffer<float, TMapping>* weightedDer,
                                 TCudaBuffer<float, TMapping>* weightedDer2,
                                 ui32 stream = 0) {
    using TKernel = NKernelHost::TPointwiseTargetImplKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(), stream, target, weights, point, alpha, lossFunction, score, weightedDer, weightedDer2);
}

template <class TMapping>
inline void ApproximateCrossEntropy(const TCudaBuffer<const float, TMapping>& target,
                                    const TCudaBuffer<const float, TMapping>& weights,
                                    const TCudaBuffer<const float, TMapping>& point,
                                    TCudaBuffer<float, TMapping>* score,
                                    TCudaBuffer<float, TMapping>* weightedDer,
                                    TCudaBuffer<float, TMapping>* weightedDer2,
                                    bool useBorder,
                                    float border,
                                    ui32 stream = 0) {
    using TKernel = NKernelHost::TCrossEntropyTargetKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(), stream, target, weights, point, score, weightedDer, weightedDer2, border, useBorder);
}

template <class TMapping>
inline void ApproximateQueryRmse(const TCudaBuffer<const ui32, TMapping>& querySizes,
                                 const TCudaBuffer<const ui32, TMapping>& queryOffsets,
                                 NCudaLib::TDistributedObject<ui32> offsetsBias,
                                 const TCudaBuffer<const float, TMapping>& target,
                                 const TCudaBuffer<const float, TMapping>& weights,
                                 const TCudaBuffer<const float, TMapping>& point,
                                 const TCudaBuffer<ui32, TMapping>* indices,
                                 TCudaBuffer<float, TMapping>* score,
                                 TCudaBuffer<float, TMapping>* weightedDer,
                                 TCudaBuffer<float, TMapping>* weightedDer2,
                                 ui32 stream = 0) {
    using TKernel = NKernelHost::TQueryRmseKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(), stream,
                           querySizes, queryOffsets, offsetsBias,
                           target, weights, point,
                           indices,
                           score, weightedDer, weightedDer2);
}

template <class TMapping>
inline void ApproximateQuerySoftMax(const TCudaBuffer<const ui32, TMapping>& querySizes,
                                    const TCudaBuffer<const ui32, TMapping>& queryOffsets,
                                    NCudaLib::TDistributedObject<ui32> offsetsBias,
                                    const TCudaBuffer<const float, TMapping>& target,
                                    const TCudaBuffer<const float, TMapping>& weights,
                                    const TCudaBuffer<const float, TMapping>& point,
                                    const TCudaBuffer<ui32, TMapping>* indices,
                                    TCudaBuffer<float, TMapping>* score,
                                    TCudaBuffer<float, TMapping>* weightedDer,
                                    TCudaBuffer<float, TMapping>* weightedDer2,
                                    ui32 stream = 0) {
    using TKernel = NKernelHost::TQuerySoftMaxKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(), stream,
                           querySizes, queryOffsets, offsetsBias,
                           target, weights, point,
                           indices,
                           score, weightedDer, weightedDer2);
}

template <class TMapping>
inline void ApproximatePairLogit(const TCudaBuffer<const uint2, TMapping>& pairs,
                                 const TCudaBuffer<const float, TMapping>& pairWeigts,
                                 NCudaLib::TDistributedObject<ui32> offsetsBias,
                                 const TCudaBuffer<const float, TMapping>& point,
                                 const TCudaBuffer<ui32, TMapping>* indices,
                                 TCudaBuffer<float, TMapping>* score,
                                 TCudaBuffer<float, TMapping>* weightedDer,
                                 TCudaBuffer<float, TMapping>* weightedDer2,
                                 ui32 stream = 0) {
    using TKernel = NKernelHost::TPairLogitKernel;
    LaunchKernels<TKernel>(pairs.NonEmptyDevices(), stream,
                           pairs, pairWeigts, offsetsBias,
                           point,
                           indices,
                           score, weightedDer, weightedDer2);
}

template <class TMapping>
inline void MakePairWeights(const TCudaBuffer<const uint2, TMapping>& pairs,
                            const TCudaBuffer<const float, TMapping>& pairWeights,
                            TCudaBuffer<float, TMapping>& weights,
                            ui32 stream = 0) {
    using TKernel = NKernelHost::TMakePairWeightsKernel;
    LaunchKernels<TKernel>(pairs.NonEmptyDevices(), stream,
                           pairs, pairWeights, weights);
}

template <class TMapping>
inline void ApproximateYetiRank(ui64 seed, ui32 permutationCount,
                                const TCudaBuffer<const ui32, TMapping>& querySizes,
                                const TCudaBuffer<const ui32, TMapping>& queryOffsets,
                                NCudaLib::TDistributedObject<ui32> offsetsBias,
                                const TCudaBuffer<const float, TMapping>& target,
                                const TCudaBuffer<const float, TMapping>& point,
                                const TCudaBuffer<ui32, TMapping>* indices,
                                TCudaBuffer<float, TMapping>* score,
                                TCudaBuffer<float, TMapping>* weightedDer,
                                TCudaBuffer<float, TMapping>* weightedDer2,
                                ui32 stream = 0) {
    using TKernel = NKernelHost::TYetiRankKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(), stream,
                           querySizes, queryOffsets, offsetsBias,
                           target, point,
                           seed, permutationCount,
                           indices,
                           score,
                           weightedDer,
                           weightedDer2);
}
