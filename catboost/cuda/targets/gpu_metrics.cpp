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

#include <catboost/libs/helpers/math_utils.h>

using namespace NCudaLib;

namespace NCatboostCuda {
    //layout: approxClass * columns + targetClass

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

        virtual TMetricHolder Eval(const TStripeBuffer<const float>& target,
                                   const TStripeBuffer<const float>& weights,
                                   const TStripeBuffer<const float>& cursor,
                                   TScopedCacheHolder* cache) const final {
            Y_UNUSED(cache);
            return EvalOnGpu<NCudaLib::TStripeMapping>(target, weights, cursor, cache);
        }

        virtual TMetricHolder Eval(const TMirrorBuffer<const float>& target,
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
            const auto& params = GetMetricDescription().GetLossParams();
            switch (metricType) {
                case ELossFunction::Logloss:
                case ELossFunction::CrossEntropy: {
                    float border = GetDefaultClassificationBorder();
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
                                            cursor,
                                            &tmp,
                                            (TVec*)nullptr,
                                            (TVec*)nullptr,
                                            useBorder,
                                            border);

                    const double sum = ReadReduce(tmp)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalWeight);
                }
                case ELossFunction::RMSE: {
                    auto tmp = TVec::CopyMapping(cursor);
                    tmp.Copy(cursor);
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
                case ELossFunction::Poisson: {
                    float alpha = 0.5;
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
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

                    ApproximatePointwise(target,
                                         weights,
                                         cursor,
                                         metricType,
                                         alpha,
                                         &tmp,
                                         (TVec*)nullptr,
                                         (TVec*)nullptr);

                    auto result = ReadReduce(tmp);
                    const double multiplier = (metricType == ELossFunction::MAE ? 2.0 : 1.0);
                    return MakeSimpleAdditiveStatistic(-result[0] * multiplier, totalWeight);
                }
                case ELossFunction::MultiClass: {
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
                    MultiLogitValueAndDer(target, weights, cursor, (const TCudaBuffer<ui32, TMapping>*)nullptr,
                                          NumClasses, &tmp, (TVec*)nullptr);
                    const double sum = ReadReduce(tmp)[0];
                    return MakeSimpleAdditiveStatistic(sum, totalWeight);
                }
                case ELossFunction::MultiClassOneVsAll: {
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
                    MultiClassOneVsAllValueAndDer(target, weights, cursor, (const TCudaBuffer<ui32, TMapping>*)nullptr,
                                                  NumClasses, &tmp, (TVec*)nullptr);
                    const double sum = ReadReduce(tmp)[0];
                    return MakeSimpleAdditiveStatistic(sum, totalWeight);
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
                BuildConfusionMatrix(target, cursor, numClasses, IsBinClass, &bins);

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

        virtual TMetricHolder Eval(const TStripeBuffer<const float>& target,
                                   const TStripeBuffer<const float>& weights,
                                   const TGpuSamplesGrouping<NCudaLib::TStripeMapping>& samplesGrouping,
                                   const TStripeBuffer<const float>& cursor) const {
            return EvalOnGpu<NCudaLib::TStripeMapping>(target, weights, samplesGrouping, cursor);
        }

        virtual TMetricHolder Eval(const TMirrorBuffer<const float>& target,
                                   const TMirrorBuffer<const float>& weights,
                                   const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& samplesGrouping,
                                   const TMirrorBuffer<const float>& cursor) const {
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
                    const auto& params = GetMetricDescription().GetLossParams();

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
                                           NPar::TLocalExecutor* localExecutor) const {
        const IMetric& metric = GetCpuMetric();
        const int start = 0;
        const int end = static_cast<const int>(metric.GetErrorType() == EErrorType::PerObjectError ? target.size() : queriesInfo.size());
        CB_ENSURE(approx.size() >= 1);
        for (ui32 dim = 0; dim < approx.size(); ++dim) {
            CB_ENSURE(approx[dim].size() == target.size());
        }
        return metric.Eval(approx,
                           target,
                           weight,
                           queriesInfo,
                           start,
                           end,
                           *localExecutor);
    }

    static TVector<THolder<IGpuMetric>> CreateGpuMetricFromDescription(ELossFunction targetObjective, const NCatboostOptions::TLossDescription& metricDescription, ui32 approxDim) {
        TVector<THolder<IGpuMetric>> result;
        const auto numClasses = approxDim == 1 ? 2 : approxDim;
        const bool isMulticlass = IsMultiClassMetric(targetObjective);
        if (isMulticlass) {
            CB_ENSURE(approxDim > 1, "Error: multiclass approx is > 1");
        } else {
            CB_ENSURE(approxDim == 1, "Error: non-multiclass output dim should be equal to  1");
        }

        auto metricType = metricDescription.GetLossFunction();
        switch (metricType) {
            case ELossFunction::Logloss:
            case ELossFunction::CrossEntropy:
            case ELossFunction::RMSE:
            case ELossFunction::Lq:
            case ELossFunction::Quantile:
            case ELossFunction::MAE:
            case ELossFunction::LogLinQuantile:
            case ELossFunction::MultiClass:
            case ELossFunction::MultiClassOneVsAll:
            case ELossFunction::MAPE:
            case ELossFunction::Accuracy:
            case ELossFunction::ZeroOneLoss:
            case ELossFunction::NumErrors:
            case ELossFunction::Poisson: {
                result.push_back(new TGpuPointwiseMetric(metricDescription, approxDim));
                break;
            }
            case ELossFunction::TotalF1: {
                result.emplace_back(new TGpuPointwiseMetric(MakeTotalF1Metric(numClasses), 0, numClasses, isMulticlass, metricDescription));
                break;
            }
            case ELossFunction::MCC: {
                result.emplace_back(new TGpuPointwiseMetric(MakeMCCMetric(numClasses), 0, numClasses, isMulticlass, metricDescription));
                break;
            }
            case ELossFunction::F1: {
                if (approxDim == 1) {
                    result.emplace_back(new TGpuPointwiseMetric(MakeBinClassF1Metric(), 1, 2, isMulticlass, metricDescription));
                } else {
                    for (ui32 i = 0; i < approxDim; ++i) {
                        result.emplace_back(new TGpuPointwiseMetric(MakeMultiClassF1Metric(i), i, approxDim, isMulticlass, metricDescription));
                    }
                }
                break;
            }
            case ELossFunction::AUC: {
                if (approxDim == 1) {
                    result.emplace_back(new TGpuPointwiseMetric(MakeBinClassAucMetric(), 1, 2, isMulticlass, metricDescription));
                } else {
                    CATBOOST_WARNING_LOG << "AUC is not implemented on GPU. Will use CPU for metric computation, this could significantly affect learning time" << Endl;
                    for (ui32 i = 0; i < approxDim; ++i) {
                        result.emplace_back(new TCpuFallbackMetric(MakeMultiClassAucMetric(i), metricDescription));
                    }
                }
                break;
            }
            case ELossFunction::Kappa: {
                if (approxDim == 1) {
                    result.emplace_back(new TCpuFallbackMetric(MakeBinClassKappaMetric(), metricDescription));
                } else {
                    result.emplace_back(new TCpuFallbackMetric(MakeMultiClassKappaMetric(numClasses), metricDescription));
                }
                break;
            }

            case ELossFunction::WKappa: {
                if (approxDim == 1) {
                    result.emplace_back(new TCpuFallbackMetric(MakeBinClassWKappaMetric(), metricDescription));
                } else {
                    result.emplace_back(new TCpuFallbackMetric(MakeMultiClassWKappaMetric(numClasses), metricDescription));
                }
                break;
            }

            case ELossFunction::HammingLoss: {
                double border = GetDefaultClassificationBorder();
                const auto& params = metricDescription.GetLossParams();
                if (params.contains("border")) {
                    border = FromString<float>(params.at("border"));
                }

                if (approxDim == 1) {
                    result.emplace_back(new TCpuFallbackMetric(MakeHammingLossMetric(border), metricDescription));
                } else {
                    result.emplace_back(new TCpuFallbackMetric(MakeHammingLossMetric(border, true), metricDescription));
                }
                break;
            }

            case ELossFunction::HingeLoss: {
                result.emplace_back(new TCpuFallbackMetric(MakeHingeLossMetric(), metricDescription));
                break;
            }

            case ELossFunction::Precision: {
                if (approxDim == 1) {
                    result.emplace_back(new TGpuPointwiseMetric(MakeBinClassPrecisionMetric(), 1, 2, isMulticlass, metricDescription));
                } else {
                    for (ui32 i = 0; i < approxDim; ++i) {
                        result.emplace_back(new TGpuPointwiseMetric(MakeMultiClassPrecisionMetric(i), i, approxDim, isMulticlass, metricDescription));
                    }
                }
                break;
            }
            case ELossFunction::Recall: {
                if (approxDim == 1) {
                    result.emplace_back(new TGpuPointwiseMetric(MakeBinClassRecallMetric(), 1, 2, isMulticlass, metricDescription));
                } else {
                    for (ui32 i = 0; i < approxDim; ++i) {
                        result.emplace_back(new TGpuPointwiseMetric(MakeMultiClassRecallMetric(i), i, approxDim, isMulticlass, metricDescription));
                    }
                }
                break;
            }
            case ELossFunction::NDCG:
            case ELossFunction::QueryRMSE:
            case ELossFunction::QuerySoftMax:
            case ELossFunction::PairLogit:
            case ELossFunction::PairLogitPairwise: {
                result.push_back(new TGpuQuerywiseMetric(metricDescription, approxDim));
                break;
            }
            case ELossFunction::QueryCrossEntropy: {
                CB_ENSURE(targetObjective == ELossFunction::QueryCrossEntropy, "Error: could compute QueryCrossEntropy metric on GPU only for QueryCrossEntropyObjective");
                result.push_back(new TTargetFallbackMetric(metricDescription, approxDim));
                break;
            }
            default: {
                CB_ENSURE(approxDim == 1, "Error: can't use CPU for unknown multiclass metric");
                THolder<IGpuMetric> metric = new TCpuFallbackMetric(metricDescription, approxDim);
                CATBOOST_WARNING_LOG << "Metric " << metric->GetCpuMetric().GetDescription() << " is not implemented on GPU. Will use CPU for metric computation, this could significantly affect learning time" << Endl;
                result.push_back(std::move(metric));
                break;
            }
        }
        return result;
    }

    TVector<THolder<IGpuMetric>> CreateGpuMetrics(const NCatboostOptions::TOption<NCatboostOptions::TLossDescription>& lossFunctionOption,
                                                  const NCatboostOptions::TOption<NCatboostOptions::TMetricOptions>& evalMetricOptions,
                                                  ui32 cpuApproxDim) {
        TVector<THolder<IGpuMetric>> metrics;
        THashSet<TString> usedDescriptions;

        if (evalMetricOptions->EvalMetric.IsSet()) {
            if (evalMetricOptions->EvalMetric->GetLossFunction() == ELossFunction::PythonUserDefinedPerObject) {
                CB_ENSURE(false, "Error: GPU doesn't support custom metrics");
            } else {
                TVector<THolder<IGpuMetric>> createdMetrics = CreateGpuMetricFromDescription(lossFunctionOption->GetLossFunction(),
                                                                                             evalMetricOptions->EvalMetric,
                                                                                             cpuApproxDim);
                CB_ENSURE(createdMetrics.size() == 1, "Eval metric should have a single value. Metric " << ToString(evalMetricOptions->EvalMetric->GetLossFunction()) << " provides a value for each class, thus it cannot be used as "
                                                                                                        << "a single value to select best iteration or to detect overfitting. "
                                                                                                        << "If you just want to look on the values of this metric use custom_metric parameter.");
                metrics.push_back(std::move(createdMetrics.front()));
                usedDescriptions.insert(metrics.back()->GetCpuMetric().GetDescription());
            }
        }

        CB_ENSURE(lossFunctionOption->GetLossFunction() != ELossFunction::PythonUserDefinedPerObject, "Error: GPU doesn't support user-defined loss");

        for (auto&& metric : CreateGpuMetricFromDescription(lossFunctionOption->GetLossFunction(),
                                                            lossFunctionOption, cpuApproxDim)) {
            const TString& description = metric->GetCpuMetric().GetDescription();
            if (!usedDescriptions.contains(description)) {
                usedDescriptions.insert(description);
                metrics.push_back(std::move(metric));
            }
        }

        for (const auto& description : evalMetricOptions->CustomMetrics.Get()) {
            for (auto&& metric : CreateGpuMetricFromDescription(lossFunctionOption->GetLossFunction(),
                                                                description, cpuApproxDim)) {
                const TString& description = metric->GetCpuMetric().GetDescription();
                if (!usedDescriptions.contains(description)) {
                    usedDescriptions.insert(description);
                    metrics.push_back(std::move(metric));
                }
            }
        }

        return metrics;
    }

}
