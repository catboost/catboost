#include "gpu_metrics.h"

#include "auc.h"
#include "dcg.h"
#include "kernel.h"
#include "multiclass_kernels.h"
#include "query_cross_entropy_kernels.h"

#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/gpu_data/kernels.h>
#include <catboost/cuda/gpu_data/querywise_helper.h>
#include <catboost/cuda/cuda_util/partitions_reduce.h>
#include <catboost/cuda/targets/user_defined.h>
#include <catboost/libs/helpers/math_utils.h>

using namespace NCudaLib;

namespace NCatboostCuda {
    //layout: approxClass * columns + targetClass

    static constexpr float GetBinTargetProbabilityThreshold() {
        return 0.5;
    }

    //target pos
    static inline double TruePositivePlusFalseNegative(const TVector<double>& confusionMatrix, ui32 numClasses,
                                                       ui32 classIdx) {
        double result = 0;
        for (ui32 approxIdx = 0; approxIdx < numClasses; ++approxIdx) {
            result += confusionMatrix[approxIdx * numClasses + classIdx];
        }
        return result;
    }

    //approx pos
    static inline double TruePositivePlusFalsePositive(const TVector<double>& confusionMatrix, ui32 numClasses,
                                                       ui32 approxIdx) {
        double result = 0;
        for (ui32 classIdx = 0; classIdx < numClasses; ++classIdx) {
            result += confusionMatrix[approxIdx * numClasses + classIdx];
        }
        return result;
    }

    static inline double TruePositive(const TVector<double>& confusionMatrix, ui32 numClasses, ui32 classIdx) {
        return confusionMatrix[classIdx * numClasses + classIdx];
    }

    static inline TMetricHolder Accuracy(const TVector<double>& confusionMatrix) {
        const ui32 numClasses = sqrt(confusionMatrix.size());
        TMetricHolder result(2);
        for (ui32 classIdx = 0; classIdx < numClasses; ++classIdx) {
            result.Stats[0] += confusionMatrix[classIdx * numClasses + classIdx];
        }
        for (double val : confusionMatrix) {
            result.Stats[1] += val;
        }
        return result;
    }

    static inline TMetricHolder Precision(const TVector<double>& confusionMatrix, ui32 classIdx) {
        const ui32 numClasses = sqrt(confusionMatrix.size());
        TMetricHolder result(2);
        result.Stats[0] = TruePositive(confusionMatrix, numClasses, classIdx);
        result.Stats[1] = TruePositivePlusFalsePositive(confusionMatrix, numClasses, classIdx);
        return result;
    }

    static inline TMetricHolder Recall(const TVector<double>& confusionMatrix, ui32 classIdx) {
        const ui32 numClasses = sqrt(confusionMatrix.size());
        TMetricHolder result(2);
        result.Stats[0] = TruePositive(confusionMatrix, numClasses, classIdx);
        result.Stats[1] = TruePositivePlusFalseNegative(confusionMatrix, numClasses, classIdx);
        return result;
    }

    static inline void BuildTotalF1Stats(const TVector<double>& confusionMatrix,
                                         TVector<double>* statsPtr) {
        const ui32 numClasses = sqrt(confusionMatrix.size());

        statsPtr->resize(numClasses * 3);
        auto& stats = *statsPtr;

        for (ui32 classIdx = 0; classIdx < numClasses; ++classIdx) {
            stats[3 * classIdx] = TruePositivePlusFalseNegative(confusionMatrix, numClasses, classIdx);
            stats[3 * classIdx + 1] = TruePositivePlusFalsePositive(confusionMatrix, numClasses, classIdx);
            stats[3 * classIdx + 2] = TruePositive(confusionMatrix, numClasses, classIdx);
        }
    }

    static inline void BuildF1Stats(const TVector<double>& confusionMatrix,
                                    ui32 classIdx,
                                    TVector<double>* statsPtr) {
        const ui32 numClasses = sqrt(confusionMatrix.size());
        statsPtr->resize(3);
        auto& stats = *statsPtr;

        stats[0] = TruePositive(confusionMatrix, numClasses, classIdx);
        stats[1] = TruePositivePlusFalseNegative(confusionMatrix, numClasses, classIdx);
        stats[2] = TruePositivePlusFalsePositive(confusionMatrix, numClasses, classIdx);
    }

    IGpuMetric::IGpuMetric(const NCatboostOptions::TLossDescription& description, ui32 approxDim)
        : CpuMetric(std::move(CreateMetricFromDescription(description, approxDim)[0]))
        , MetricDescription(description)
    {
    }

    IGpuMetric::IGpuMetric(THolder<IMetric>&& cpuMetric, const NCatboostOptions::TLossDescription& description)
        : CpuMetric(std::move(cpuMetric))
        , MetricDescription(description)
    {
    }

    static inline TMetricHolder MakeSimpleAdditiveStatistic(double sum, double weight) {
        TMetricHolder stat(2);
        stat.Stats[0] = sum;
        stat.Stats[1] = weight;
        return stat;
    }

    template <class T, class TMapping>
    static inline double SumVector(const TCudaBuffer<T, TMapping>& vec) {
        using TVec = TCudaBuffer<std::remove_const_t<T>, TMapping>;
        auto tmp = TVec::CopyMapping(vec);
        FillBuffer(tmp, 1.0f);
        return DotProduct(tmp, vec);
    }

    class TGpuPointwiseMetric: public IGpuPointwiseMetric, public TGuidHolder {
    public:
        explicit TGpuPointwiseMetric(const NCatboostOptions::TLossDescription& config, ui32 approxDim)
            : IGpuPointwiseMetric(config, approxDim)
            , NumClasses(approxDim == 1 ? 2 : approxDim)
            , ClassIdx(1)
            , IsBinClass(approxDim == 1)
        {
        }

        explicit TGpuPointwiseMetric(THolder<IMetric>&& cpuMetric, ui32 classIdx, ui32 numClasses, bool isMulticlass, const NCatboostOptions::TLossDescription& config)
            : IGpuPointwiseMetric(std::move(cpuMetric), config)
            , NumClasses(numClasses)
            , ClassIdx(isMulticlass ? classIdx : 1)
            , IsBinClass(!isMulticlass)

        {
        }

        TMetricHolder Eval(const TStripeBuffer<const float>& target,
                                   const TStripeBuffer<const float>& weights,
                                   const TStripeBuffer<const float>& cursor,
                                   TScopedCacheHolder* cache) const final {
            Y_UNUSED(cache);
            return EvalOnGpu<NCudaLib::TStripeMapping>(target, weights, cursor, cache);
        }

        TMetricHolder Eval(const TMirrorBuffer<const float>& target,
                                   const TMirrorBuffer<const float>& weights,
                                   const TMirrorBuffer<const float>& cursor,
                                   TScopedCacheHolder* cache) const final {
            return EvalOnGpu<NCudaLib::TMirrorMapping>(target, weights, cursor, cache);
        }

    private:
        template <class TMapping>
        TMetricHolder EvalOnGpu(const TCudaBuffer<const float, TMapping>& target,
                                const TCudaBuffer<const float, TMapping>& weights,
                                const TCudaBuffer<const float, TMapping>& cursor,
                                TScopedCacheHolder* cache) const {
            using TVec = TCudaBuffer<float, TMapping>;

            double totalWeight = SumVector(weights);
            auto metricType = GetMetricDescription().GetLossFunction();
            const auto& params = GetMetricDescription().GetLossParamsMap();
            // for models with uncertainty
            // compatibility of metrics and loss is checked at startup in CheckMetric
            const auto& prediction0 = cursor.ColumnView(0);
            switch (metricType) {
                case ELossFunction::Logloss:
                case ELossFunction::CrossEntropy: {
                    float border = GetDefaultTargetBorder();
                    bool useBorder = false;
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
                    if (metricType == ELossFunction::Logloss) {
                        useBorder = true;
                        if (params.contains("border")) {
                            border = FromString<float>(params.at("border"));
                        }
                    }

                    ApproximateCrossEntropy(target,
                                            weights,
                                            prediction0,
                                            &tmp,
                                            (TVec*)nullptr,
                                            (TVec*)nullptr,
                                            useBorder,
                                            border);

                    const double sum = ReadReduce(tmp)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalWeight);
                }
                case ELossFunction::RMSE: {
                    auto tmp = TVec::CopyMapping(prediction0);
                    tmp.Copy(prediction0);
                    SubtractVector(tmp, target);
                    const double sum2 = DotProduct(tmp, tmp, &weights);
                    return MakeSimpleAdditiveStatistic(sum2, totalWeight);
                }
                case ELossFunction::Quantile:
                case ELossFunction::MAE:
                case ELossFunction::LogLinQuantile:
                case ELossFunction::Lq:
                case ELossFunction::NumErrors:
                case ELossFunction::MAPE:
                case ELossFunction::Poisson:
                case ELossFunction::Expectile:
                case ELossFunction::Tweedie:
                case ELossFunction::Huber: {
                    float alpha = 0.5;
                    auto tmp = TVec::Create(prediction0.GetMapping().RepeatOnAllDevices(1));
                    //TODO(noxoomo): make param dispatch on device side
                    if (params.contains("alpha")) {
                        alpha = FromString<float>(params.at("alpha"));
                    }
                    if (metricType == ELossFunction::NumErrors) {
                        alpha = FromString<float>(params.at("greater_than"));
                    }
                    if (metricType == ELossFunction::Lq) {
                        alpha = FromString<float>(params.at("q"));
                    }
                    if (metricType == ELossFunction::Tweedie) {
                        alpha = FromString<float>(params.at("variance_power"));
                    }
                    if (metricType == ELossFunction::Huber) {
                        alpha = FromString<float>(params.at("delta"));
                    }

                    ApproximatePointwise(target,
                                         weights,
                                         prediction0,
                                         metricType,
                                         alpha,
                                         &tmp,
                                         (TVec*)nullptr,
                                         (TVec*)nullptr);

                    auto result = ReadReduce(tmp);
                    const double multiplier = (metricType == ELossFunction::MAE ? 2.0 : 1.0);
                    return MakeSimpleAdditiveStatistic(-result[0] * multiplier, totalWeight);
                }
                case ELossFunction::RMSEWithUncertainty: {
                    CB_ENSURE(NumClasses == 2, "Expect two-dimensional predictions");
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
                    RMSEWithUncertaintyValueAndDer(target, weights, cursor, (const TCudaBuffer<ui32, TMapping>*)nullptr,
                                          &tmp, (TVec*)nullptr);
                    const double sum = ReadReduce(tmp)[0];
                    CB_ENSURE(IsFinite(sum), "Numerical overflow in RMSEWithUncertainty");
                    return MakeSimpleAdditiveStatistic(-sum, totalWeight);
                }
                case ELossFunction::MultiClass: {
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
                    MultiLogitValueAndDer(target, weights, cursor, (const TCudaBuffer<ui32, TMapping>*)nullptr,
                                          NumClasses, &tmp, (TVec*)nullptr);
                    const double sum = ReadReduce(tmp)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalWeight);
                }
                case ELossFunction::MultiClassOneVsAll: {
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
                    MultiClassOneVsAllValueAndDer(target, weights, cursor, (const TCudaBuffer<ui32, TMapping>*)nullptr,
                                                  NumClasses, &tmp, (TVec*)nullptr);
                    const double sum = ReadReduce(tmp)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalWeight);
                }
                case ELossFunction::MultiCrossEntropy:
                case ELossFunction::MultiLogloss: {
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
                    MultiCrossEntropyValueAndDer(
                        target,
                        weights,
                        cursor,
                        (const TCudaBuffer<ui32, TMapping>*)nullptr,
                        &tmp,
                        (TVec*)nullptr);
                    const double sum = ReadReduce(tmp)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalWeight);
                }
                case ELossFunction::MultiRMSE: {
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
                    MultiRMSEValueAndDer(
                        target,
                        weights,
                        cursor,
                        (const TCudaBuffer<ui32, TMapping>*)nullptr,
                        &tmp,
                        (TVec*)nullptr);
                    const double sum = ReadReduce(tmp)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalWeight);
                }
                case ELossFunction::MCC: {
                    return BuildConfusionMatrixAtPoint(target, weights, cursor, NumClasses, cache);
                }
                case ELossFunction::TotalF1: {
                    const auto& confusionMatrix = BuildConfusionMatrixAtPoint(target, weights, cursor, NumClasses, cache);
                    TMetricHolder result;
                    BuildTotalF1Stats(confusionMatrix.Stats, &result.Stats);
                    return result;
                }
                case ELossFunction::F1: {
                    const auto& confusionMatrix = BuildConfusionMatrixAtPoint(target, weights, cursor, NumClasses, cache);
                    TMetricHolder result;
                    BuildF1Stats(confusionMatrix.Stats, ClassIdx, &result.Stats);
                    return result;
                }
                case ELossFunction::Accuracy:
                case ELossFunction::ZeroOneLoss: {
                    return Accuracy(BuildConfusionMatrixAtPoint(target, weights, cursor, NumClasses, cache).Stats);
                }
                case ELossFunction::Recall: {
                    return Recall(BuildConfusionMatrixAtPoint(target, weights, cursor, NumClasses, cache).Stats, ClassIdx);
                }
                case ELossFunction::Precision: {
                    return Precision(BuildConfusionMatrixAtPoint(target, weights, cursor, NumClasses, cache).Stats, ClassIdx);
                }
                case ELossFunction::AUC: {
                    TMetricHolder metric(2);
                    metric.Stats[0] = ComputeAUC(target, weights, cursor);
                    metric.Stats[1] = 1;
                    return metric;
                }
                default: {
                    CB_ENSURE(false, "Unsupported on GPU pointwise metric " << metricType);
                }
            }
        }

        template <class TMapping>
        const TMetricHolder& BuildConfusionMatrixAtPoint(const TCudaBuffer<const float, TMapping>& target,
                                                         const TCudaBuffer<const float, TMapping>& weights,
                                                         const TCudaBuffer<const float, TMapping>& cursor,
                                                         ui32 numClasses,
                                                         TScopedCacheHolder* cache) const {
            return cache->Cache(*this, 0, [&]() -> TMetricHolder {
                auto indices = TCudaBuffer<ui32, TMapping>::CopyMapping(target);
                MakeSequence(indices);

                auto bins = TCudaBuffer<ui32, TMapping>::CopyMapping(target);
                BuildConfusionMatrix(target, cursor, numClasses, IsBinClass, GetBinTargetProbabilityThreshold(), &bins);

                const ui32 matrixSize = numClasses * numClasses;
                ReorderBins(bins, indices, 0, NCB::IntLog2(matrixSize));

                TCudaBuffer<ui32, TMapping> offsets;
                offsets.Reset(target.GetMapping().RepeatOnAllDevices(matrixSize + 1));
                UpdatePartitionOffsets(bins, offsets);

                auto tempWeights = TCudaBuffer<float, TMapping>::CopyMapping(weights);
                Gather(tempWeights, weights, indices);
                auto stats = TCudaBuffer<double, TMapping>::Create(target.GetMapping().RepeatOnAllDevices(matrixSize));
                ComputePartitionStats(tempWeights, offsets, &stats);
                TMetricHolder holder;
                holder.Stats = ReadReduce(stats);
                CB_ENSURE(holder.Stats.size() == matrixSize);
                return holder;
            });
        }

    private:
        ui32 NumClasses = 0;
        ui32 ClassIdx = 1;
        bool IsBinClass = true;
    };

    class TGpuQuerywiseMetric: public IGpuQuerywiseMetric {
    public:
        explicit TGpuQuerywiseMetric(const NCatboostOptions::TLossDescription& config, ui32 approxDim)
            : IGpuQuerywiseMetric(config, approxDim)
        {
        }

        TMetricHolder Eval(const TStripeBuffer<const float>& target,
                                   const TStripeBuffer<const float>& weights,
                                   const TGpuSamplesGrouping<NCudaLib::TStripeMapping>& samplesGrouping,
                                   const TStripeBuffer<const float>& cursor) const override {
            return EvalOnGpu<NCudaLib::TStripeMapping>(target, weights, samplesGrouping, cursor);
        }

        TMetricHolder Eval(const TMirrorBuffer<const float>& target,
                                   const TMirrorBuffer<const float>& weights,
                                   const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& samplesGrouping,
                                   const TMirrorBuffer<const float>& cursor) const override {
            return EvalOnGpu<NCudaLib::TMirrorMapping>(target,
                                                       weights,
                                                       samplesGrouping,
                                                       cursor);
        }

    private:
        template <class TMapping>
        TMetricHolder EvalOnGpu(const TCudaBuffer<const float, TMapping>& target,
                                const TCudaBuffer<const float, TMapping>& weights,
                                const TGpuSamplesGrouping<TMapping>& samplesGrouping,
                                const TCudaBuffer<const float, TMapping>& cursor) const {
            using TVec = TCudaBuffer<float, TMapping>;
            auto value = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
            FillBuffer(value, 0.0f);

            auto metricType = GetMetricDescription().GetLossFunction();
            switch (metricType) {
                case ELossFunction::QueryRMSE: {
                    double totalWeight = SumVector(weights);
                    ApproximateQueryRmse(samplesGrouping.GetSizes(),
                                         samplesGrouping.GetBiasedOffsets(),
                                         samplesGrouping.GetOffsetsBias(),
                                         target,
                                         weights,
                                         cursor,
                                         (TCudaBuffer<ui32, TMapping>*)nullptr,
                                         &value,
                                         (TVec*)nullptr,
                                         (TVec*)nullptr);
                    double sum = ReadReduce(value)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalWeight);
                }
                case ELossFunction::QuerySoftMax: {
                    double totalWeightedTarget = DotProduct(target,
                                                            weights);
                    ApproximateQuerySoftMax(samplesGrouping.GetSizes(),
                                            samplesGrouping.GetBiasedOffsets(),
                                            samplesGrouping.GetOffsetsBias(),
                                            NCatboostOptions::GetQuerySoftMaxLambdaReg(GetMetricDescription()),
                                            NCatboostOptions::GetQuerySoftMaxBeta(GetMetricDescription()),
                                            target,
                                            weights,
                                            cursor,
                                            (TCudaBuffer<ui32, TMapping>*)nullptr,
                                            &value,
                                            (TVec*)nullptr,
                                            (TVec*)nullptr);
                    double sum = ReadReduce(value)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalWeightedTarget);
                }
                case ELossFunction::PairLogitPairwise:
                case ELossFunction::PairLogit: {
                    double totalPairsWeight = SumVector(samplesGrouping.GetPairsWeights());
                    ApproximatePairLogit(samplesGrouping.GetPairs(),
                                         samplesGrouping.GetPairsWeights(),
                                         samplesGrouping.GetOffsetsBias(),
                                         cursor,
                                         (TCudaBuffer<ui32, TMapping>*)nullptr,
                                         &value,
                                         (TVec*)nullptr,
                                         (TVec*)nullptr);
                    double sum = ReadReduce(value)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalPairsWeight);
                }
                case ELossFunction::NDCG: {
                    const auto& params = GetMetricDescription().GetLossParamsMap();

                    auto type = ENdcgMetricType::Base;
                    if (const auto it = params.find("type"); it != params.end()) {
                        type = FromString<ENdcgMetricType>(it->second);
                    }

                    auto top = Max<ui32>();
                    if (const auto it = params.find("top"); it != params.end()) {
                        top = FromString<ui32>(it->second);
                    }

                    // TODO(yazevnul): we can compute multiple NDCG metrics with different `top`
                    // parameter (but `type` must be the same) in one function call.
                    const auto perQueryNdcgSum = CalculateNdcg(
                                                     samplesGrouping.GetSizes(),
                                                     samplesGrouping.GetBiasedOffsets(),
                                                     samplesGrouping.GetOffsetsBias(),
                                                     weights,
                                                     target,
                                                     cursor,
                                                     type,
                                                     {top})
                                                     .front();
                    auto queryWeights = TCudaBuffer<float, TMapping>::CopyMapping(samplesGrouping.GetSizes());
                    NDetail::GatherBySizeAndOffset(
                        weights,
                        samplesGrouping.GetSizes(),
                        samplesGrouping.GetBiasedOffsets(),
                        samplesGrouping.GetOffsetsBias(),
                        queryWeights,
                        1);
                    const auto queryWeightsSum = ReduceToHost(queryWeights);
                    return MakeSimpleAdditiveStatistic(perQueryNdcgSum, queryWeightsSum);
                }
                default: {
                    CB_ENSURE(false, "Unsupported on GPU querywise metric " << metricType);
                }
            }
        }
    };

    TMetricHolder TCpuFallbackMetric::Eval(const TVector<TVector<double>>& approx,
                                           const TVector<float>& target,
                                           const TVector<float>& weight,
                                           const TVector<TQueryInfo>& queriesInfo,
                                           NPar::ILocalExecutor* localExecutor) const {
        const int start = 0;
        const int end = static_cast<const int>(GetCpuMetric().GetErrorType() == EErrorType::PerObjectError ? target.size() : queriesInfo.size());
        CB_ENSURE(approx.size() >= 1);
        CB_ENSURE(end > 0, "Not enough data to calculate metric: groupwise metric w/o group id's, or objectwise metric w/o samples");
        for (ui32 dim = 0; dim < approx.size(); ++dim) {
            CB_ENSURE(approx[dim].size() == target.size());
        }
        const ISingleTargetEval& singleEvalMetric = dynamic_cast<const ISingleTargetEval&>(GetCpuMetric());
        return singleEvalMetric.Eval(approx,
                           target,
                           weight,
                           queriesInfo,
                           start,
                           end,
                           *localExecutor);
    }

    TMetricHolder TGpuCustomMetric::Eval(
        const TStripeBuffer<const float>& target,
        const TStripeBuffer<const float>& weights,
        const TStripeBuffer<const float>& cursor,
        TScopedCacheHolder* cache,
        ui32 stream
    ) const {
        return EvalImpl<TStripeMapping>(target, weights, cursor, cache, stream);
    }

    TMetricHolder TGpuCustomMetric::Eval(
        const TMirrorBuffer<const float>& target,
        const TMirrorBuffer<const float>& weights,
        const TMirrorBuffer<const float>& cursor,
        TScopedCacheHolder* cache,
        ui32 stream
    ) const {
        return EvalImpl<TMirrorMapping>(target, weights, cursor, cache, stream);
    }

    template<class TMapping>
    TMetricHolder TGpuCustomMetric::EvalImpl(
        const TCudaBuffer<const float, TMapping>& target,
        const TCudaBuffer<const float, TMapping>& weights,
        const TCudaBuffer<const float, TMapping>& cursor,
        TScopedCacheHolder* cache,
        ui32 stream
    ) const {
        using TKernel = NKernelHost::TUserDefinedMetricKernel;
        using TVec = TCudaBuffer<float, TMapping>;
        Y_UNUSED(cache);

        const size_t tmpArraySize = TKernel::BlockSize * TKernel::NumBlocks;
        auto resultTmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(tmpArraySize));
        auto resultWeightsTmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(tmpArraySize));

        auto onesTmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(tmpArraySize));
        FillBuffer(onesTmp, 1.0f);

        LaunchKernels<TKernel>(target.NonEmptyDevices(), stream, target, weights, cursor, resultTmp, resultWeightsTmp, Descriptor);
        return MakeSimpleAdditiveStatistic(DotProduct(resultTmp, onesTmp), DotProduct(resultWeightsTmp, onesTmp));
    }

    double TGpuCustomMetric::GetFinalError(TMetricHolder &&metricHolder) const {
        return (*(Descriptor.GetFinalErrorFunc))(metricHolder, Descriptor.CustomData);
    }

    static THolder<IMetric> CreateSingleMetric(ELossFunction metric, const TLossParams& params, int approxDimension) {
        THolder<IMetric> metricHolder = std::move(CreateMetric(metric, params, approxDimension)[0]);
        return metricHolder;
    }

    static TVector<THolder<IGpuMetric>> CreateGpuMetricFromDescription(ELossFunction targetObjective, const NCatboostOptions::TLossDescription& metricDescription, ui32 approxDim) {
        TVector<THolder<IGpuMetric>> result;
        const bool isMulticlass = IsMultiClassOnlyMetric(targetObjective);
        const bool isMultiLabel = IsMultiLabelObjective(targetObjective);
        const bool isRMSEWithUncertainty = targetObjective == ELossFunction::RMSEWithUncertainty;
        const bool isMultiRMSE = targetObjective == ELossFunction::MultiRMSE;
        if (isMulticlass || isMultiLabel || isRMSEWithUncertainty || isMultiRMSE) {
            CB_ENSURE(approxDim > 1, "Error: Approx dimension equal to 1 for multidimensional output");
        } else {
            CB_ENSURE(approxDim == 1, "Error: non-multidimensional output dim should be equal to 1");
        }

        auto metricType = metricDescription.GetLossFunction();
        const TLossParams& params = metricDescription.GetLossParams();

        switch (metricType) {
            case ELossFunction::Logloss:
            case ELossFunction::CrossEntropy:
            case ELossFunction::RMSE:
            case ELossFunction::RMSEWithUncertainty:
            case ELossFunction::Lq:
            case ELossFunction::Quantile:
            case ELossFunction::MAE:
            case ELossFunction::LogLinQuantile:
            case ELossFunction::MultiClass:
            case ELossFunction::MultiClassOneVsAll:
            case ELossFunction::MultiCrossEntropy:
            case ELossFunction::MultiLogloss:
            case ELossFunction::MultiRMSE:
            case ELossFunction::MAPE:
            case ELossFunction::Accuracy:
            case ELossFunction::ZeroOneLoss:
            case ELossFunction::NumErrors:
            case ELossFunction::TotalF1:
            case ELossFunction::MCC:
            case ELossFunction::Poisson:
            case ELossFunction::Expectile:
            case ELossFunction::Tweedie:
            case ELossFunction::Huber: {
                result.emplace_back(new TGpuPointwiseMetric(metricDescription, approxDim));
                break;
            }
            case ELossFunction::Precision:
            case ELossFunction::Recall:
            case ELossFunction::F1: {
                auto cpuMetrics = CreateMetricFromDescription(metricDescription, approxDim);
                const auto numClasses = approxDim == 1 ? 2 : approxDim;
                for (ui32 i = 0; i < approxDim; ++i) {
                    result.emplace_back(new TGpuPointwiseMetric(std::move(cpuMetrics[i]), i, numClasses, isMulticlass || isMultiLabel, metricDescription));
                }
                break;
            }
            case ELossFunction::AUC: {
                auto cpuMetrics = CreateMetricFromDescription(metricDescription, approxDim);
                if ((approxDim == 1) && (IsClassificationObjective(targetObjective) || targetObjective == ELossFunction::QueryCrossEntropy)) {
                    CB_ENSURE_INTERNAL(
                        cpuMetrics.size() == 1,
                        "CreateMetricFromDescription for AUC for binclass should return one-element vector"
                    );
                    result.emplace_back(new TGpuPointwiseMetric(std::move(cpuMetrics[0]), 1, 2, isMulticlass, metricDescription));
                } else {
                    CATBOOST_WARNING_LOG << "AUC is not implemented on GPU. Will use CPU for metric computation, this could significantly affect learning time" << Endl;

                    for (auto& cpuMetric : cpuMetrics) {
                        result.emplace_back(new TCpuFallbackMetric(std::move(cpuMetric), metricDescription));
                    }
                }
                break;
            }
            case ELossFunction::Kappa: {
                if (approxDim == 1) {
                    result.emplace_back(new TCpuFallbackMetric(CreateSingleMetric(metricType, params, approxDim), metricDescription));
                } else {
                    result.emplace_back(new TCpuFallbackMetric(CreateSingleMetric(metricType, params, approxDim), metricDescription));
                }
                break;
            }

            case ELossFunction::WKappa: {
                if (approxDim == 1) {
                    result.emplace_back(new TCpuFallbackMetric(CreateSingleMetric(metricType, params, approxDim), metricDescription));
                } else {
                    result.emplace_back(new TCpuFallbackMetric(CreateSingleMetric(metricType, params, approxDim), metricDescription));
                }
                break;
            }

            case ELossFunction::HammingLoss: {
                result.emplace_back(new TCpuFallbackMetric(CreateSingleMetric(metricType, params, approxDim), metricDescription));
                break;
            }

            case ELossFunction::HingeLoss: {
                result.emplace_back(new TCpuFallbackMetric(CreateSingleMetric(metricType, params, approxDim), metricDescription));
                break;
            }
            case ELossFunction::QueryRMSE:
            case ELossFunction::QuerySoftMax:
            case ELossFunction::PairLogit:
            case ELossFunction::PairLogitPairwise: {
                result.emplace_back(new TGpuQuerywiseMetric(metricDescription, approxDim));
                break;
            }
            case ELossFunction::Combination:
            case ELossFunction::QueryCrossEntropy: {
                CB_ENSURE(
                    targetObjective == ELossFunction::QueryCrossEntropy || targetObjective == ELossFunction::Combination,
                    "Error: metric " << metricType << " on GPU requires loss function QueryCrossEntropy or Combination");
                result.emplace_back(new TTargetFallbackMetric(metricDescription, approxDim));
                break;
            }
            default: {
                CB_ENSURE(approxDim == 1, "Error: can't use CPU for unknown multiclass metric");
                THolder<IGpuMetric> metric = MakeHolder<TCpuFallbackMetric>(metricDescription, approxDim);
                CATBOOST_WARNING_LOG << "Metric " << metric->GetCpuMetric().GetDescription() << " is not implemented on GPU. Will use CPU for metric computation, this could significantly affect learning time" << Endl;
                result.push_back(std::move(metric));
                break;
            }
        }
        return result;
    }

    static inline bool ShouldConsiderWeightsByDefault(const THolder<IGpuMetric>& metric) {
        return metric->GetCpuMetric().GetDescription() != "AUC" && !metric->GetUseWeights().IsUserDefined() &&
               !metric->GetUseWeights().IsIgnored();
    }

    TVector<THolder<IGpuMetric>> CreateGpuMetrics(
        const NCatboostOptions::TOption<NCatboostOptions::TMetricOptions>& metricOptions,
        ui32 cpuApproxDim,
        bool hasWeights,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor
    ) {
        CB_ENSURE(metricOptions->ObjectiveMetric.IsSet(), "Objective metric must be set.");
        const NCatboostOptions::TLossDescription& objectiveMetricDescription = metricOptions->ObjectiveMetric.Get();
        const bool haveEvalMetricFromUser = metricOptions->EvalMetric.IsSet();
        const NCatboostOptions::TLossDescription& evalMetricDescription =
                haveEvalMetricFromUser ? metricOptions->EvalMetric.Get() : objectiveMetricDescription;

        TVector<THolder<IGpuMetric>> createdObjectiveMetrics;
        if (!IsUserDefined(objectiveMetricDescription.GetLossFunction())){
            createdObjectiveMetrics = CreateGpuMetricFromDescription(
                objectiveMetricDescription.GetLossFunction(),
                objectiveMetricDescription,
                cpuApproxDim);
        }

        if (hasWeights) {
            for (auto&& metric : createdObjectiveMetrics) {
                const auto& useWeights = metric->GetUseWeights();
                if (!useWeights.IsIgnored() && !useWeights.IsUserDefined()){
                    metric->GetUseWeights().SetDefaultValue(true);
                }
            }
        }

        TVector<THolder<IGpuMetric>> metrics;
        THashSet<TString> usedDescriptions;

        if (IsUserDefined(evalMetricDescription.GetLossFunction())) {
            if (evalMetricDescriptor.Defined() && evalMetricDescriptor.GetRef().GpuEvalFunc.Defined()) {
                metrics.emplace_back(new TGpuCustomMetric(
                    evalMetricDescriptor.GetRef(),
                    evalMetricDescription
                ));
            } else {
                metrics.emplace_back(new TCpuFallbackMetric(
                    MakeCustomMetric(evalMetricDescriptor.GetRef()),
                    evalMetricDescription
                ));
            }
        } else {
            metrics = CreateGpuMetricFromDescription(
                    objectiveMetricDescription.GetLossFunction(),
                    evalMetricDescription,
                    cpuApproxDim);
        }
        CB_ENSURE(metrics.size() == 1, "Eval metric should have a single value. Metric "
                << ToString(objectiveMetricDescription.GetLossFunction())
                << " provides a value for each class, thus it cannot be used as "
                << "a single value to select best iteration or to detect overfitting. "
                << "If you just want to look on the values of this metric use custom_metric parameter.");

        if (hasWeights && !metrics.back()->GetUseWeights().IsIgnored()) {
            if (!haveEvalMetricFromUser) {
                metrics.back()->GetUseWeights() = createdObjectiveMetrics.back()->GetUseWeights();
            } else if (ShouldConsiderWeightsByDefault(metrics.back())) {
                metrics.back()->GetUseWeights().SetDefaultValue(true);
                CATBOOST_INFO_LOG << "Note: eval_metric is using sample weights by default. " <<
                                  "Set MetricName:use_weights=False to calculate unweighted metric." << Endl;
            }
        }
        usedDescriptions.insert(metrics.back()->GetCpuMetric().GetDescription());

        for (auto&& metric : createdObjectiveMetrics) {
            const TString& description = metric->GetCpuMetric().GetDescription();
            if (!usedDescriptions.contains(description)) {
                usedDescriptions.insert(description);
                metrics.push_back(std::move(metric));
            }
        }
        // if custom metric is set without 'use_weights' parameter and we have non-default weights, we calculate both versions of metric.
        for (const auto& description : metricOptions->CustomMetrics.Get()) {
            TVector<THolder<IGpuMetric>> createdCustomMetrics =
                    CreateGpuMetricFromDescription(metricOptions->ObjectiveMetric->GetLossFunction(),
                                                   description,
                                                   cpuApproxDim);
            if (hasWeights) {
                TVector<THolder<IGpuMetric>> createdCustomMetricsCopy =
                        CreateGpuMetricFromDescription(metricOptions->ObjectiveMetric->GetLossFunction(),
                                                       description,
                                                       cpuApproxDim);
                auto iter = createdCustomMetricsCopy.begin();
                ui32 initialVectorSize = createdCustomMetrics.size();
                for (ui32 ind = 0; ind < initialVectorSize; ++ind) {
                    auto& metric = createdCustomMetrics[ind];
                    if (ShouldConsiderWeightsByDefault(metric)) {
                        metric->GetUseWeights() = true;
                        (*iter)->GetUseWeights() = false;
                        createdCustomMetrics.emplace_back(std::move(*iter));
                    }
                    ++iter;
                }
            }
            for (auto&& metric : createdCustomMetrics) {
                const TString& metricDescription = metric->GetCpuMetric().GetDescription();
                if (!usedDescriptions.contains(metricDescription)) {
                    usedDescriptions.insert(metricDescription);
                    metrics.push_back(std::move(metric));
                }
            }
        }

        if (!hasWeights) {
            for (const auto& metric : metrics) {
                CB_ENSURE(!metric->GetUseWeights().IsUserDefined(),
                          "If non-default weights for objects are not set, the 'use_weights' parameter must not be specified.");
            }
        }
        return metrics;
    }
}
