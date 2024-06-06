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
#include <catboost/cuda/targets/kernel/pfound_f.cuh>
#include <catboost/cuda/gpu_data/kernel/query_helper.cuh>
#include <catboost/cuda/targets/user_defined.h>

#include <util/generic/cast.h>

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
            if (FunctionValue.Size()) {
                NKernel::FillBuffer(FunctionValue.Get(), 0.0f, 1, stream.GetStream());
            }
            if (Predictions.Size() == 0) {
                return;
            }
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
            context->QueryMeans = memoryManager.Allocate<float>(QuerySizes.Size());
            context->Qids = memoryManager.Allocate<ui32>(Relevs.Size());
            context->MseDer = memoryManager.Allocate<float>(Relevs.Size());
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
            if (FunctionValue.Size()) {
                NKernel::FillBuffer(FunctionValue.Get(), 0.0f, 1, stream.GetStream());
            }
            if (Predictions.Size() == 0) {
                return;
            }

            if (Der.Size()) {
                CB_ENSURE(Der.Size() == Predictions.Size());
            }
            CB_ENSURE(QuerySizes.Size() == QueryOffsets.Size());
            if (Indices.Size()) {
                CB_ENSURE(Indices.Size() == Predictions.Size());
                NKernel::Gather(context.MseDer.Get(), Predictions.Get(), Indices.Get(), Indices.Size(), stream.GetStream());
            } else {
                CopyMemoryAsync(Predictions.Get(), context.MseDer.Get(), Predictions.Size(), stream);
            }

            NKernel::MultiplyVector(context.MseDer.Get(), -1.0f, Predictions.Size(), stream.GetStream());
            NKernel::AddVector(context.MseDer.Get(), Relevs.Get(), Relevs.Size(), stream.GetStream());
            NKernel::ComputeGroupMeans(context.MseDer.Get(), Weights.Get(), QueryOffsets.Get(), QueryOffsetsBias,
                                       QuerySizes.Get(), QueryOffsets.Size(), context.QueryMeans,
                                       stream.GetStream());
            NKernel::ComputeGroupIds(QuerySizes.Get(), QueryOffsets.Get(), QueryOffsetsBias, QueryOffsets.Size(),
                                     context.Qids, stream.GetStream());
            NKernel::ApproximateQueryRmse(context.MseDer.Get(),
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
        float LambdaReg;
        float Beta;

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
                          QueryOffsets, QuerySizes, QueryOffsetsBias, LambdaReg, Beta,
                          Indices, FunctionValue,
                          Der, Der2);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();

            context->ApproxExp = memoryManager.Allocate<float>(Relevs.Size());
            context->QueryApprox = memoryManager.Allocate<float>(QuerySizes.Size());
            context->QuerySumWeightedTargets = memoryManager.Allocate<float>(QuerySizes.Size());
            context->Qids = memoryManager.Allocate<ui32>(Relevs.Size());
            return context;
        }

        TQuerySoftMaxKernel() = default;

        TQuerySoftMaxKernel(TCudaBufferPtr<const ui32> querySizes,
                            TCudaBufferPtr<const ui32> queryOffsets,
                            ui32 queryOffsetsBias,
                            float lambdaReg,
                            float beta,
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
            , LambdaReg(lambdaReg)
            , Beta(beta)
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
            if (FunctionValue.Size()) {
                NKernel::FillBuffer(FunctionValue.Get(), 0.0f, 1, stream.GetStream());
            }
            if (Predictions.Size() == 0) {
                return;
            }

            if (Der.Size()) {
                CB_ENSURE(Der.Size() == Predictions.Size());
            }
            CB_ENSURE(QuerySizes.Size() == QueryOffsets.Size());
            if (Indices.Size()) {
                CB_ENSURE(Indices.Size() == Predictions.Size());
                NKernel::Gather(context.ApproxExp.Get(), Predictions.Get(), Indices.Get(), Indices.Size(), stream.GetStream());
            } else {
                CopyMemoryAsync(Predictions.Get(), context.ApproxExp.Get(), Predictions.Size(), stream);
            }

            NKernel::ComputeGroupIds(QuerySizes.Get(), QueryOffsets.Get(), QueryOffsetsBias, QueryOffsets.Size(), context.Qids.Get(), stream.GetStream());
            NKernel::ComputeGroupMaximals(Relevs.Get(),
                                          Weights.Get(),
                                          context.ApproxExp.Get(),
                                          QueryOffsets.Get(),
                                          QueryOffsetsBias,
                                          QuerySizes.Get(),
                                          QueryOffsets.Size(),
                                          context.QueryApprox.Get(),
                                          context.QuerySumWeightedTargets.Get(),
                                          stream.GetStream());
            NKernel::ComputeQueryExponents(Weights.Get(),
                                           context.Qids.Get(),
                                           static_cast<ui32>(Predictions.Size()),
                                           context.QueryApprox.Get(),
                                           Indices.Get(),
                                           context.ApproxExp.Get(),
                                           Beta,
                                           stream.GetStream());
            NKernel::ComputeGroupSums(context.ApproxExp.Get(),
                                      QueryOffsets.Get(),
                                      QueryOffsetsBias,
                                      QuerySizes.Get(),
                                      QueryOffsets.Size(),
                                      context.QueryApprox.Get(),
                                      stream.GetStream());

            NKernel::ApproximateQuerySoftMax(Relevs.Get(),
                                             Weights.Get(),
                                             context.ApproxExp.Get(),
                                             context.Qids.Get(),
                                             LambdaReg,
                                             Beta,
                                             static_cast<ui32>(Predictions.Size()),
                                             context.QueryApprox.Get(),
                                             context.QuerySumWeightedTargets.Get(),
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
        TCudaBufferPtr<const float> QuerywiseWeights;
        TCudaBufferPtr<const float> Predictions;

        ui64 Seed;
        ui32 PermutationCount;
        float Decay;

        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;
        TCudaBufferPtr<float> Der2;

    public:
        using TKernelContext = NKernel::TYetiRankContext;
        Y_SAVELOAD_DEFINE(QueryOffsets, QuerySizes, QueryOffsetsBias,
                          Relevs, QuerywiseWeights, Predictions,
                          Seed, PermutationCount, Decay,
                          Indices, FunctionValue, Der, Der2);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            //now ptrs would not change until kernel is finished
            context->QueryMeans = memoryManager.Allocate<float>(QuerySizes.Size());
            context->Qids = memoryManager.Allocate<ui32>(Relevs.Size());
            context->LastProceededQid = memoryManager.Allocate<ui32>(1);
            context->Approxes = memoryManager.Allocate<float>(Relevs.Size());

            if (Indices.Size()) {
                context->TempDers = memoryManager.Allocate<float>(Relevs.Size());
                context->TempWeights = memoryManager.Allocate<float>(Relevs.Size());
            }
            return context;
        }

        TYetiRankKernel() = default;

        TYetiRankKernel(TCudaBufferPtr<const ui32> querySizes,
                        TCudaBufferPtr<const ui32> queryOffsets,
                        ui32 queryOffsetsBias,
                        TCudaBufferPtr<const float> relevs,
                        TCudaBufferPtr<const float> querywiseWeights,
                        TCudaBufferPtr<const float> predictions,
                        ui64 seed, ui32 permutationCount, float decay,
                        TCudaBufferPtr<const ui32> indices,
                        TCudaBufferPtr<float> functionValue,
                        TCudaBufferPtr<float> der,
                        TCudaBufferPtr<float> der2)
            : QuerySizes(querySizes)
            , QueryOffsets(queryOffsets)
            , QueryOffsetsBias(queryOffsetsBias)
            , Relevs(relevs)
            , QuerywiseWeights(querywiseWeights)
            , Predictions(predictions)
            , Seed(seed)
            , PermutationCount(permutationCount)
            , Decay(decay)
            , Indices(indices)
            , FunctionValue(functionValue)
            , Der(der)
            , Der2(der2)
        {
        }

        void Run(const TCudaStream& stream, TKernelContext& context) {
            CB_ENSURE(Der.Size() == Predictions.Size());
            CB_ENSURE(Der2.Size() == Predictions.Size());

            if (FunctionValue.Size()) {
                NKernel::FillBuffer(FunctionValue.Get(), 0.0f, 1, stream.GetStream());
            }
            if (Predictions.Size() == 0) {
                return;
            }

            float* derDst;
            float* weightsDst;

            CB_ENSURE(QuerySizes.Size() == QueryOffsets.Size());
            if (Indices.Size()) {
                CB_ENSURE(Indices.Size() == Predictions.Size());
                NKernel::Gather(context.Approxes.Get(), Predictions.Get(), Indices.Get(), Indices.Size(), stream.GetStream());
                derDst = context.TempDers.Get();
                weightsDst = context.TempWeights.Get();
            } else {
                CopyMemoryAsync(Predictions.Get(), context.Approxes.Get(), Predictions.Size(), stream);
                derDst = Der.Get();
                weightsDst = Der2.Get();
            }

            //we adjust target by group means to avoid exponents with of relatively big numbers
            NKernel::ComputeGroupMeans(context.Approxes.Get(), nullptr, QueryOffsets.Get(), QueryOffsetsBias, QuerySizes.Get(), QueryOffsets.Size(), context.QueryMeans.Get(), stream.GetStream());
            NKernel::ComputeGroupIds(QuerySizes.Get(), QueryOffsets.Get(), QueryOffsetsBias, QueryOffsets.Size(), context.Qids.Get(), stream.GetStream());
            NKernel::RemoveQueryMeans((int*)(context.Qids.Get()), QuerySizes.Size(), context.QueryMeans.Get(), context.Approxes.Get(), stream.GetStream());

            NKernel::YetiRankGradient(Seed, Decay,
                                      PermutationCount,
                                      QueryOffsets.Get(),
                                      (int*)context.LastProceededQid.Get(),
                                      QueryOffsetsBias,
                                      QueryOffsets.Size(),
                                      (int*)context.Qids.Get(),
                                      context.Approxes.Get(),
                                      Relevs.Get(),
                                      QuerywiseWeights.Get(),
                                      Predictions.Size(),
                                      derDst,
                                      weightsDst,
                                      stream.GetStream());

            if (Indices.Size()) {
                NKernel::Scatter(Der.Get(), context.TempDers.Get(), Indices.Get(), Der.Size(), stream.GetStream());
                NKernel::Scatter(Der2.Get(), context.TempWeights.Get(), Indices.Get(), Der.Size(), stream.GetStream());
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
        Y_SAVELOAD_DEFINE(Pairs, PairWeights, QueryOffsetsBias, Predictions, Indices, FunctionValue, Der, Der2);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            if (Indices.Get()) {
                context->GatheredPoint = memoryManager.Allocate<float>(Indices.Size());
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
                NKernel::Gather(context.GatheredPoint.Get(), Predictions.Get(), Indices.Get(), Indices.Size(), stream.GetStream());
                point = context.GatheredPoint.Get();
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
                                              SafeIntegerCast<ui32>(Pairs.Size()), QueryOffsetsBias,
                                              FunctionValue.Get(),
                                              Der.Get(),
                                              Der2.Get(),
                                              docCount,
                                              stream.GetStream());
        }
    };

    class TPairLogitPairwiseKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Point;
        TCudaBufferPtr<const uint2> Pairs;
        TCudaBufferPtr<const float> PairWeights;
        TCudaBufferPtr<const ui32> ScatterDerIndices;

        TCudaBufferPtr<float> Func;
        TCudaBufferPtr<float> PointDer;
        TCudaBufferPtr<float> PairDer2;

    public:
        Y_SAVELOAD_DEFINE(Point, Pairs, PairWeights, ScatterDerIndices, Func, PointDer, PairDer2);

        TPairLogitPairwiseKernel() = default;

        TPairLogitPairwiseKernel(TCudaBufferPtr<const float> point,
                                 TCudaBufferPtr<const uint2> pairs,
                                 TCudaBufferPtr<const float> pairWeights,
                                 TCudaBufferPtr<const ui32> scatterDerIndices,
                                 TCudaBufferPtr<float> func,
                                 TCudaBufferPtr<float> der,
                                 TCudaBufferPtr<float> der2)
            : Point(point)
            , Pairs(pairs)
            , PairWeights(pairWeights)
            , ScatterDerIndices(scatterDerIndices)
            , Func(func)
            , PointDer(der)
            , PairDer2(der2)
        {
        }

        void Run(const TCudaStream& stream) {
            CB_ENSURE(Point.Size() == PointDer.Size());
            CB_ENSURE(Pairs.Size() == PairDer2.Size() || PairDer2.Size() == 0);

            NKernel::PairLogitPairwise(Point.Get(),
                                       Pairs.Get(),
                                       PairWeights.Get(),
                                       ScatterDerIndices.Get(),
                                       Func.Get(),
                                       PointDer.Get(),
                                       PointDer.Size(),
                                       PairDer2.Get(),
                                       SafeIntegerCast<ui32>(Pairs.Size()),
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

    class TComputeMatrixSizesKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> QueryOffsets;
        TCudaBufferPtr<ui32> MatrixSize;

    public:
        TComputeMatrixSizesKernel() = default;

        TComputeMatrixSizesKernel(TCudaBufferPtr<const ui32> queryOffsets,
                                  TCudaBufferPtr<ui32> matrixSize)
            : QueryOffsets(queryOffsets)
            , MatrixSize(matrixSize)
        {
        }

        Y_SAVELOAD_DEFINE(QueryOffsets,
                          MatrixSize);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(QueryOffsets.Size() > 0);
            const ui32 qCount = QueryOffsets.Size() - 1;
            CB_ENSURE(MatrixSize.Size() == QueryOffsets.Size());
            NKernel::ComputeMatrixSizes(QueryOffsets.Get(),
                                        qCount,
                                        MatrixSize.Get(),
                                        stream.GetStream());
        }
    };

    class TMakePairsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> QOffsets;
        TCudaBufferPtr<const ui64> MatrixOffsets;
        TCudaBufferPtr<uint2> Pairs;

    public:
        TMakePairsKernel() = default;

        TMakePairsKernel(TCudaBufferPtr<const ui32> qOffsets,
                         TCudaBufferPtr<const ui64> matrixOffsets,
                         TCudaBufferPtr<uint2> pairs)
            : QOffsets(qOffsets)
            , MatrixOffsets(matrixOffsets)
            , Pairs(pairs)
        {
        }

        Y_SAVELOAD_DEFINE(QOffsets, MatrixOffsets, Pairs);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(QOffsets.Size() > 0, "Need some query offsets");
            CB_ENSURE(QOffsets.Size() == MatrixOffsets.Size(), "Number of query offsets and matrix offsets should be same");
            const ui32 queryCount = QOffsets.Size() - 1;

            NKernel::MakePairs(QOffsets.Get(),
                               MatrixOffsets.Get(),
                               queryCount,
                               Pairs.Get(),
                               stream.GetStream());
        }
    };

    class TPFoundFGradientKernel: public TKernelBase<NKernel::TPFoundFContext, false> {
    private:
        ui64 Seed;
        float DecaySpeed;
        ui32 BootstrapIter;
        TCudaBufferPtr<const ui32> Qids;
        TCudaBufferPtr<const ui32> QueryOffsets;
        TCudaBufferPtr<const ui64> MatrixOffsets;
        TCudaBufferPtr<const float> ExpApprox;
        TCudaBufferPtr<const float> Relev;
        TCudaBufferPtr<float> WeightMatrixDst;

    public:
        using TKernelContext = NKernel::TPFoundFContext;

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            context->QidCursor = memoryManager.Allocate<ui32, NCudaLib::EPtrType::CudaDevice>(1);
            return context;
        }

        TPFoundFGradientKernel() = default;

        TPFoundFGradientKernel(ui64 seed, float decaySpeed, ui32 bootstrapIter, TCudaBufferPtr<const ui32> qids, TCudaBufferPtr<const ui32> queryOffsets, TCudaBufferPtr<const ui64> matrixOffsets, TCudaBufferPtr<const float> expApprox, TCudaBufferPtr<const float> relev, TCudaBufferPtr<float> weightMatrixDst)
            : Seed(seed)
            , DecaySpeed(decaySpeed)
            , BootstrapIter(bootstrapIter)
            , Qids(qids)
            , QueryOffsets(queryOffsets)
            , MatrixOffsets(matrixOffsets)
            , ExpApprox(expApprox)
            , Relev(relev)
            , WeightMatrixDst(weightMatrixDst)
        {
        }

        Y_SAVELOAD_DEFINE(Seed, DecaySpeed, BootstrapIter, QueryOffsets, Qids, MatrixOffsets, ExpApprox, Relev, WeightMatrixDst);

        void Run(const TCudaStream& stream,
                 TKernelContext& context) const {
            CB_ENSURE(QueryOffsets.Size() > 0, "Need some query offsets");
            const ui32 queryCount = QueryOffsets.Size() - 1;
            NKernel::PFoundFGradient(Seed, DecaySpeed, BootstrapIter,
                                     QueryOffsets.Get(),
                                     context.QidCursor, queryCount,
                                     Qids.Get(), MatrixOffsets.Get(), ExpApprox.Get(), Relev.Get(), Relev.Size(), WeightMatrixDst.Get(), stream.GetStream());
        }
    };

    class TMakeFinalTargetKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> DocIds;
        TCudaBufferPtr<const float> ExpApprox;
        TCudaBufferPtr<const float> Relevs;
        TCudaBufferPtr<const float> QuerywiseWeights;
        TCudaBufferPtr<float> NzPairWeights;
        TCudaBufferPtr<float> ResultDers;
        TCudaBufferPtr<uint2> NzPairs;

    public:
        TMakeFinalTargetKernel() = default;

        TMakeFinalTargetKernel(TCudaBufferPtr<const ui32> docIds,
                               TCudaBufferPtr<const float> expApprox,
                               TCudaBufferPtr<const float> querywiseWeights,
                               TCudaBufferPtr<const float> relevs,
                               TCudaBufferPtr<float> nzPairWeights,
                               TCudaBufferPtr<float> resultDers,
                               TCudaBufferPtr<uint2> nzPairs)
            : DocIds(docIds)
            , ExpApprox(expApprox)
            , Relevs(relevs)
            , QuerywiseWeights(querywiseWeights)
            , NzPairWeights(nzPairWeights)
            , ResultDers(resultDers)
            , NzPairs(nzPairs)
        {
        }

        Y_SAVELOAD_DEFINE(DocIds, ExpApprox, QuerywiseWeights, Relevs, NzPairWeights, ResultDers, NzPairs);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(NzPairWeights.Size() == NzPairs.Size());
            NKernel::MakeFinalTarget(DocIds.Get(), ExpApprox.Get(), QuerywiseWeights.Get(), Relevs.Get(), NzPairWeights.Get(), SafeIntegerCast<ui32>(NzPairWeights.Size()), ResultDers.Get(), NzPairs.Get(), stream.GetStream());
        }
    };

    class TSwapWrongOrderPairsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Relevs;
        TCudaBufferPtr<uint2> NzPairs;

    public:
        TSwapWrongOrderPairsKernel() = default;

        TSwapWrongOrderPairsKernel(TCudaBufferPtr<const float> relevs,
                                   TCudaBufferPtr<uint2> nzPairs)
            : Relevs(relevs)
            , NzPairs(nzPairs)
        {
        }

        Y_SAVELOAD_DEFINE(Relevs, NzPairs);

        void Run(const TCudaStream& stream) const {
            NKernel::SwapWrongOrderPairs(Relevs.Get(), SafeIntegerCast<ui32>(NzPairs.Size()), NzPairs.Get(), stream.GetStream());
        }
    };

    class TRemoveOffsetsBias: public TStatelessKernel {
    private:
        ui32 Bias;
        TCudaBufferPtr<uint2> NzPairs;

    public:
        TRemoveOffsetsBias() = default;

        TRemoveOffsetsBias(ui32 bias, TCudaBufferPtr<uint2> nzPairs)
            : Bias(bias)
            , NzPairs(nzPairs)
        {
        }

        Y_SAVELOAD_DEFINE(NzPairs, Bias);

        void Run(const TCudaStream& stream) const {
            NKernel::RemoveOffsetsBias(Bias, SafeIntegerCast<ui32>(NzPairs.Size()), NzPairs.Get(), stream.GetStream());
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
                                 ELossFunction lossFunction,
                                 float alpha,
                                 TCudaBuffer<float, TMapping>* score,
                                 TCudaBuffer<float, TMapping>* weightedDer,
                                 TCudaBuffer<float, TMapping>* weightedDer2,
                                 ui32 stream = 0) {
    using TKernel = NKernelHost::TPointwiseTargetImplKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(), stream, target, weights, point, alpha, lossFunction, score, weightedDer, weightedDer2);
}

template <class TMapping>
inline void ApproximateUserDefined(const TCudaBuffer<const float, TMapping>& target,
                                   const TCudaBuffer<const float, TMapping>& weights,
                                   const TCudaBuffer<const float, TMapping>& point,
                                   const TCustomObjectiveDescriptor& objectiveDescriptor,
                                   TCudaBuffer<float, TMapping>* value,
                                   TCudaBuffer<float, TMapping>* weightedDer,
                                   TCudaBuffer<float, TMapping>* weightedDer2,
                                   ui32 stream = 0) {
    using TKernel = NKernelHost::TUserDefinedObjectiveKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(), stream, objectiveDescriptor, target, weights, point, value, weightedDer, weightedDer2);
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
                                    float lambdaReg,
                                    float beta,
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
                           lambdaReg, beta, target, weights, point,
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
inline void ApproximateYetiRank(ui64 seed, float decay, ui32 permutationCount,
                                const TCudaBuffer<const ui32, TMapping>& querySizes,
                                const TCudaBuffer<const ui32, TMapping>& queryOffsets,
                                NCudaLib::TDistributedObject<ui32> offsetsBias,
                                const TCudaBuffer<const float, TMapping>& target,
                                const TCudaBuffer<const float, TMapping>& querywiseWeights,
                                const TCudaBuffer<const float, TMapping>& point,
                                const TCudaBuffer<ui32, TMapping>* indices,
                                TCudaBuffer<float, TMapping>* score,
                                TCudaBuffer<float, TMapping>* weightedDer,
                                TCudaBuffer<float, TMapping>* weightedDer2,
                                ui32 stream = 0) {
    using TKernel = NKernelHost::TYetiRankKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(), stream,
                           querySizes, queryOffsets, offsetsBias,
                           target,
                           querywiseWeights,
                           point,
                           seed,
                           permutationCount, decay,
                           indices,
                           score,
                           weightedDer,
                           weightedDer2);
}

template <class TMapping>
inline void MakePairs(const TCudaBuffer<ui32, TMapping>& qidOffsets,
                      const TCudaBuffer<ui64, TMapping>& matrixOffsets,
                      TCudaBuffer<uint2, TMapping>* pairs,
                      ui32 stream = 0) {
    using TKernel = NKernelHost::TMakePairsKernel;
    LaunchKernels<TKernel>(qidOffsets.NonEmptyDevices(), stream, qidOffsets, matrixOffsets, *pairs);
}

template <class TMapping>
inline void ComputeMatrixSizes(const TCudaBuffer<ui32, TMapping>& qidOffsets,
                               TCudaBuffer<ui32, TMapping>* matrixSizes,
                               ui32 stream = 0) {
    using TKernel = NKernelHost::TComputeMatrixSizesKernel;
    LaunchKernels<TKernel>(qidOffsets.NonEmptyDevices(), stream, qidOffsets, *matrixSizes);
}

inline void ComputePFoundFWeightsMatrix(NCudaLib::TDistributedObject<ui64> seed,
                                        float decaySpeed,
                                        ui32 permutationCount,
                                        const TCudaBuffer<float, NCudaLib::TStripeMapping>& expApprox,
                                        const TCudaBuffer<float, NCudaLib::TStripeMapping>& target,
                                        const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& qids,
                                        const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& qidOffsets,
                                        const TCudaBuffer<ui64, NCudaLib::TStripeMapping>& matrixOffsets,
                                        TCudaBuffer<float, NCudaLib::TStripeMapping>* weights,
                                        ui32 stream = 0) {
    using TKernel = NKernelHost::TPFoundFGradientKernel;
    LaunchKernels<TKernel>(qidOffsets.NonEmptyDevices(),
                           stream,
                           seed,
                           decaySpeed,
                           permutationCount,
                           qids,
                           qidOffsets,
                           matrixOffsets,
                           expApprox,
                           target,
                           *weights);
}

inline void MakeFinalPFoundGradients(const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& docs,
                                     const TCudaBuffer<float, NCudaLib::TStripeMapping>& expApprox,
                                     const TCudaBuffer<float, NCudaLib::TStripeMapping>& querywiseWeights,
                                     const TCudaBuffer<float, NCudaLib::TStripeMapping>& target,
                                     TCudaBuffer<float, NCudaLib::TStripeMapping>* pairWeights,
                                     TCudaBuffer<uint2, NCudaLib::TStripeMapping>* pairs,
                                     TCudaBuffer<float, NCudaLib::TStripeMapping>* gradient,
                                     ui32 stream = 0) {
    using TKernel = NKernelHost::TMakeFinalTargetKernel;
    LaunchKernels<TKernel>(pairs->NonEmptyDevices(),
                           stream,
                           docs,
                           expApprox,
                           querywiseWeights,
                           target,
                           *pairWeights,
                           *gradient,
                           *pairs);
}

inline void SwapWrongOrderPairs(const TCudaBuffer<const float, NCudaLib::TStripeMapping>& target,
                                TCudaBuffer<uint2, NCudaLib::TStripeMapping>* pairs,
                                i32 stream = 0) {
    using TKernel = NKernelHost::TSwapWrongOrderPairsKernel;
    LaunchKernels<TKernel>(pairs->NonEmptyDevices(),
                           stream,
                           target,
                           pairs);
}

inline void RemoveOffsetsBias(NCudaLib::TDistributedObject<ui32> bias,
                              TCudaBuffer<uint2, NCudaLib::TStripeMapping>* pairs,
                              ui32 stream = 0) {
    using TKernel = NKernelHost::TRemoveOffsetsBias;
    LaunchKernels<TKernel>(pairs->NonEmptyDevices(),
                           stream,
                           bias,
                           pairs);
}

template <class TMapping>
inline void PairLogitPairwise(const TCudaBuffer<const float, TMapping>& point,
                              const TCudaBuffer<uint2, TMapping>& pairs,
                              const TCudaBuffer<float, TMapping>& pairWeights,
                              const TCudaBuffer<ui32, TMapping>& scatterDerIndices,
                              TCudaBuffer<float, TMapping>* func,
                              TCudaBuffer<float, TMapping>* weightedPointDer,
                              TCudaBuffer<float, TMapping>* weightedPairDer2,
                              ui32 stream = 0) {
    using TKernel = NKernelHost::TPairLogitPairwiseKernel;
    LaunchKernels<TKernel>(pairs.NonEmptyDevices(),
                           stream,
                           point,
                           pairs,
                           pairWeights,
                           scatterDerIndices,
                           func,
                           weightedPointDer,
                           weightedPairDer2);
}

template <class TMapping>
inline void PairLogitPairwise(const TCudaBuffer<const float, TMapping>& point,
                              const TCudaBuffer<const uint2, TMapping>& pairs,
                              const TCudaBuffer<const float, TMapping>& pairWeights,
                              TCudaBuffer<float, TMapping>* weightedPointDer,
                              TCudaBuffer<float, TMapping>* weightedPairDer2,
                              ui32 stream = 0) {
    using TKernel = NKernelHost::TPairLogitPairwiseKernel;
    LaunchKernels<TKernel>(pairs.NonEmptyDevices(),
                           stream,
                           point,
                           pairs,
                           pairWeights,
                           static_cast<TCudaBuffer<const ui32, TMapping>*>(nullptr),
                           static_cast<TCudaBuffer<float, TMapping>*>(nullptr),
                           weightedPointDer,
                           weightedPairDer2);
}
