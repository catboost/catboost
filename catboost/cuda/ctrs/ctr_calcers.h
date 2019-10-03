#pragma once

#include "ctr_kernels.h"
#include "ctr_bins_builder.h"

#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/scan.h>
#include <catboost/cuda/cuda_util/segmented_scan.h>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/cuda_lib/cache.h>
#include <catboost/cuda/cuda_util/reduce.h>
#include <catboost/cuda/cuda_util/compression_helpers_gpu.h>

#include <catboost/private/libs/ctr_description/ctr_config.h>

namespace NCatboostCuda {



    template <class T>
    inline TVector<T> SingletonVector(const T& value) {
        return {value};
    }

    template <class TMapping>
    using TCtrVisitor = std::function<void(const NCB::TCtrConfig&, const TCudaBuffer<float, TMapping>&, ui32 stream)>;

    template <class TMapping>
    class THistoryBasedCtrCalcer: public TGuidHolder {
    public:
        using TVisitor = TCtrVisitor<TMapping>;

        template <class TFloat, class TUint32>
        THistoryBasedCtrCalcer(const TCudaBuffer<TFloat, TMapping>& weights,
                               const TCudaBuffer<TUint32, TMapping>& indices,
                               const TCudaBuffer<const ui32, TMapping>* groupIds,
                               ui32 mask = TCtrBinBuilder<TMapping>::GetMask(),
                               ui32 stream = 0)
            : TGuidHolder()
            , Indices(indices.ConstCopyView())
            , Mask(mask)
            , Stream(stream)
        {

            Reset(weights, indices, groupIds);
        }

        template <class TUint32>
        THistoryBasedCtrCalcer(const TCudaBuffer<TUint32, TMapping>& indices,
                               ui32 firstTestIndex,
                               const TCudaBuffer<const ui32, TMapping>* groupIds,
                               ui32 mask = TCtrBinBuilder<TMapping>::GetMask(),
                               ui32 stream = 0)
            : TGuidHolder()
            , Indices(indices.ConstCopyView())
            , Mask(mask)
            , Stream(stream)
        {
            Reset(indices, firstTestIndex, groupIds);
        }

        template <class TFloat, class TUint32>
        THistoryBasedCtrCalcer& Reset(const TCudaBuffer<TFloat, TMapping>& weights,
                                      const TCudaBuffer<TUint32, TMapping>& indices,
                                      const TCudaBuffer<const ui32, TMapping>* groupIds
                                      ) {
            ReserveMemoryUpdateIndicesAndMaybeCreateGroupIdsFix(indices, groupIds);


            GatherWithMask(GatheredWeightsWithMask, weights, Indices, Mask, Stream);
            WriteFloatMask(Indices, GatheredWeightsWithMask, Stream);
            SegmentedScanAndScatterNonNegativeVector(GatheredWeightsWithMask, Indices, ScannedScatteredWeights, false,
                                                     Stream);

            if (NeedFixForGroupwiseCtr()) {
                FixGroupwiseCtr(ScannedScatteredWeights);
            }

            IsBinarizedSampleWasGathered = false;
            return *this;
        }

        template <class TUint32>
        THistoryBasedCtrCalcer& Reset(const TCudaBuffer<TUint32, TMapping>& indices,
                                      const ui32 firstZeroIndex,
                                      const TCudaBuffer<const ui32, TMapping>* groupIds) {
            ReserveMemoryUpdateIndicesAndMaybeCreateGroupIdsFix(indices, groupIds);

            GatherTrivialWeights(GatheredWeightsWithMask, Indices, firstZeroIndex, true, Stream);
            SegmentedScanAndScatterNonNegativeVector(GatheredWeightsWithMask, Indices, ScannedScatteredWeights, false,
                                                     Stream);
            if (NeedFixForGroupwiseCtr()) {
                FixGroupwiseCtr(ScannedScatteredWeights);
            }
            IsBinarizedSampleWasGathered = false;
            return *this;
        }

        bool HasBinarizedTargetSample() const {
            return static_cast<bool>(BinarizedSample.GetObjectsSlice().Size());
        }

        bool HasFloatTargetSample() const {
            return static_cast<bool>(WeightedSample.GetObjectsSlice().Size());
        }

        THistoryBasedCtrCalcer& SetBinarizedSample(TCudaBuffer<const ui8, TMapping>&& binarizedSample) {
            BinarizedSample = std::move(binarizedSample);
            IsBinarizedSampleWasGathered = false;
            return *this;
        }

        THistoryBasedCtrCalcer& SetFloatSample(TCudaBuffer<const float, TMapping>&& floatSample) {
            CB_ENSURE(WeightedSample.GetObjectsSlice().Size() == 0);
            WeightedSample = std::move(floatSample);
            return *this;
        }

        THistoryBasedCtrCalcer& VisitCatFeatureCtr(const TVector<NCB::TCtrConfig>& ctrConfigs,
                                                   TVisitor& visitor) {
            CB_ENSURE(BinarizedSample.GetObjectsSlice().Size() == Indices.GetObjectsSlice().Size());
            const auto& referenceCtrConfig = ctrConfigs[0];
            CB_ENSURE(referenceCtrConfig.Type == ECtrType::Borders || referenceCtrConfig.Type == ECtrType::Buckets);
            const auto& gatheredSample = GetGatheredBinSample();

            Dst.Reset(ScannedScatteredWeights.GetMapping());
            Tmp.Reset(Dst.GetMapping());

            {
                auto profileGuard = NCudaLib::GetCudaManager().GetProfiler().Profile("compute ctr stats");

                FillBinarizedTargetsStats(gatheredSample, GatheredWeightsWithMask, Dst, referenceCtrConfig.ParamId,
                                          referenceCtrConfig.Type, Stream);
                SegmentedScanAndScatterNonNegativeVector(Dst, Indices, Tmp, false, Stream);
            }
            if (NeedFixForGroupwiseCtr()) {
                FixGroupwiseCtr(Tmp);
            }

            for (auto& ctrConfig : ctrConfigs) {
                CB_ENSURE(IsEqualUpToPriorAndBinarization(ctrConfig, referenceCtrConfig));
                const float firstClassPriorCount = GetNumeratorShift(ctrConfig);
                const float totalPriorCount = GetDenumeratorShift(ctrConfig);
                DivideWithPriors(Tmp, ScannedScatteredWeights, firstClassPriorCount, totalPriorCount, Dst, Stream);

                const auto& constDst = Dst;
                visitor(ctrConfig, constDst, Stream);
            }

            return *this;
        }

        THistoryBasedCtrCalcer& VisitCatFeatureCtr(const NCB::TCtrConfig& ctrConfig,
                                                   TVisitor& visitor) {
            return VisitCatFeatureCtr(SingletonVector(ctrConfig), visitor);
        }

        THistoryBasedCtrCalcer& ComputeCatFeatureCtr(const NCB::TCtrConfig& ctrConfig,
                                                     TCudaBuffer<float, TMapping>& dst) {
            TVisitor ctrVisitor = [&](const NCB::TCtrConfig& config,
                                      const TCudaBuffer<float, TMapping>& ctr,
                                      ui32 streamId) {
                Y_UNUSED(config);
                dst.Reset(ctr.GetMapping());
                dst.Copy(ctr, streamId);
            };
            VisitCatFeatureCtr(ctrConfig, ctrVisitor);
            return *this;
        }

        THistoryBasedCtrCalcer& VisitFloatFeatureMeanCtrs(const TVector<NCB::TCtrConfig>& ctrConfigs,
                                                          TVisitor& visitor) {
            CB_ENSURE(WeightedSample.GetObjectsSlice().Size() == Indices.GetObjectsSlice().Size());
            CB_ENSURE(ctrConfigs[0].Type == ECtrType::FloatTargetMeanValue);

            Dst.Reset(WeightedSample.GetMapping());
            Tmp.Reset(Dst.GetMapping());
            GatherWithMask(Tmp, WeightedSample, Indices, Mask, Stream);
            SegmentedScanVector(Tmp, Indices, Dst, false, 1u << 31, Stream);
            ScatterWithMask(Tmp, Dst, Indices, Mask, Stream);
            if (NeedFixForGroupwiseCtr()) {
                FixGroupwiseCtr(Tmp);
            }
            for (auto& ctrConfig : ctrConfigs) {
                CB_ENSURE(ctrConfig.Prior.size() == 2, "Error: float mean ctr need 2 priors");
                CB_ENSURE(IsEqualUpToPriorAndBinarization(ctrConfig, ctrConfigs[0]));

                const float priorSum = GetNumeratorShift(ctrConfig);
                const float priorWeight = GetDenumeratorShift(ctrConfig);
                DivideWithPriors(Tmp, ScannedScatteredWeights, priorSum, priorWeight, Dst, Stream);

                const auto& constDst = Dst;
                visitor(ctrConfig, constDst, Stream);
            }

            return *this;
        }

        THistoryBasedCtrCalcer& VisitFloatFeatureMeanCtr(const NCB::TCtrConfig& ctrConfig,
                                                         TVisitor& visitor) {
            return VisitFloatFeatureMeanCtrs(SingletonVector(ctrConfig), visitor);
        }

        THistoryBasedCtrCalcer& ComputeFloatFeatureMean(const NCB::TCtrConfig& ctrConfig,
                                                        TCudaBuffer<float, TMapping>& dst) {
            TVisitor ctrVisitor = [&](const NCB::TCtrConfig& config,
                                      const TCudaBuffer<float, TMapping>& ctr,
                                      ui32 stream) {
                Y_UNUSED(config);
                dst.Reset(ctr.GetMapping());
                dst.Copy(ctr, stream);
            };
            VisitFloatFeatureMeanCtr(ctrConfig, ctrVisitor);
            return *this;
        }

    private:

        bool NeedFixForGroupwiseCtr() const {
            return static_cast<bool>(FixForGroupwiseCtrs.GetObjectsSlice().Size());
        }

        void FixGroupwiseCtr(TCudaBuffer<float, TMapping>& ctr) const {
            ApplyFixForGroupwiseCtr(FixForGroupwiseCtrs, ctr);
        }

        template <class TUi32>
        void ReserveMemoryUpdateIndicesAndMaybeCreateGroupIdsFix(const TCudaBuffer<TUi32, TMapping>& indices_,
                                                                 const TCudaBuffer<const ui32, TMapping>* groupIds
            ) {

            auto indices = indices_.ConstCopyView();
            ScannedScatteredWeights.Reset(indices.GetMapping());
            Tmp.Reset(indices.GetMapping());
            GatheredWeightsWithMask.Reset(indices.GetMapping());
            Indices = indices.ConstCopyView();
            if (groupIds) {
                FixForGroupwiseCtrs.Reset(indices.GetMapping());
                //hacks to compute groupwise ctrs without implicit GPU sync
                auto tmp = Tmp.template ReinterpretCast<ui32>();
                auto bins = GatheredWeightsWithMask.template ReinterpretCast<ui32>();
                auto binIndices = ScannedScatteredWeights.template ReinterpretCast<ui32>();
                FillBuffer(bins, 0, Stream);
                //mark start of group for each bin
                MakeGroupStartFlags(indices, *groupIds, &tmp, Mask, Stream);
                //compute unique bin
                ScanVector(tmp, bins, false, Stream);
                //save start of group run index in binIndices
                FillBinIndices(Mask, indices, bins, &binIndices, Stream);
                CreateFixedIndices(bins, binIndices, indices, Mask, &FixForGroupwiseCtrs, Stream);
            }
        }

        const TCudaBuffer<ui8, TMapping>& GetGatheredBinSample() {
            if (!IsBinarizedSampleWasGathered) {
                GatheredBinarizedSample.Reset(BinarizedSample.GetMapping());
                GatherWithMask(GatheredBinarizedSample, BinarizedSample, Indices, Mask, Stream);
                IsBinarizedSampleWasGathered = true;
            }
            return GatheredBinarizedSample;
        }

    private:
        TCudaBuffer<const ui32, TMapping> Indices;


        TCudaBuffer<float, TMapping> Dst;
        TCudaBuffer<float, TMapping> ScannedScatteredWeights;
        TCudaBuffer<float, TMapping> Tmp;
        TCudaBuffer<float, TMapping> GatheredWeightsWithMask;
        TCudaBuffer<ui32, TMapping> FixForGroupwiseCtrs;

        TCudaBuffer<ui8, TMapping> GatheredBinarizedSample;
        bool IsBinarizedSampleWasGathered = false;

        TCudaBuffer<const float, TMapping> WeightedSample;
        TCudaBuffer<const ui8, TMapping> BinarizedSample;

        ui32 Mask;
        ui32 Stream;
    };

    template <class TMapping>
    class TWeightedBinFreqCalcer {
    public:
        using TVisitor = TCtrVisitor<TMapping>;

        template <class TFloat>
        TWeightedBinFreqCalcer(const TCudaBuffer<TFloat, TMapping>& weights,
                               float totalWeight,
                               ui32 mask = TCtrBinBuilder<TMapping>::GetMask(),
                               ui32 stream = 0)
            : Weights(weights.ConstCopyView())
            , TotalWeight(totalWeight)
            , Mask(mask)
            , Stream(stream)
        {
            //reserve memory
            SegmentStarts.Reset(weights.GetMapping());
            BinWeights.Reset(weights.GetMapping());
        }

        TWeightedBinFreqCalcer& VisitEqualUpToPriorFreqCtrs(const TCudaBuffer<const ui32, TMapping>& indices,
                                                            const TVector<NCB::TCtrConfig>& ctrConfigs,
                                                            TVisitor& visitor) {
            //TODO(noxoomo): change tempFlags to ui8
            TempFlags.Reset(indices.GetMapping());
            ExtractMask(indices, TempFlags, false, Stream);

            Bins.Reset(indices.GetMapping());
            ScanVector(TempFlags, Bins, false, Stream);
            //+1 is for one fake bin to store last segment end
            ui32 binCountWithFakeLastBin = ReadLast(Bins, Stream) + 2;

            SegmentStarts.Reset(Bins.GetMapping().RepeatOnAllDevices(binCountWithFakeLastBin));
            UpdatePartitionOffsets(Bins, SegmentStarts, Stream);

            BinWeights.Reset(Bins.GetMapping().RepeatOnAllDevices(binCountWithFakeLastBin - 1));
            Tmp.Reset(Weights.GetMapping());

            GatherWithMask(Tmp, Weights, indices, Mask, Stream);
            SegmentedReduceVector(Tmp, SegmentStarts, BinWeights, EOperatorType::Sum, Stream);

            //TODO(noxoomo): we don't calc several priors for featureFreq. btw, this part could be optimized for several priors
            for (auto& ctrConfig : ctrConfigs) {
                CB_ENSURE(ctrConfig.Type == ECtrType::FeatureFreq);
                CB_ENSURE(ctrConfig.Prior.size() == 2);
                const float prior = GetNumeratorShift(ctrConfig);
                const float observationsPrior = GetDenumeratorShift(ctrConfig);

                ComputeWeightedBinFreqCtr(indices, Bins, BinWeights, TotalWeight, prior, observationsPrior, Tmp,
                                          Stream);
                visitor(ctrConfig, Tmp, Stream);
            }
            return *this;
        }

        TWeightedBinFreqCalcer& VisitEqualUpToPriorFreqCtrs(const TCudaBuffer<ui32, TMapping>& indices,
                                                            const TVector<NCB::TCtrConfig>& ctrConfigs,
                                                            TVisitor& visitor) {
            return VisitEqualUpToPriorFreqCtrs(indices.ConstCopyView(), ctrConfigs, visitor);
        }

        template <class TUint32>
        TWeightedBinFreqCalcer& VisitFreqCtrs(const TCudaBuffer<TUint32, TMapping>& indices,
                                              const NCB::TCtrConfig& ctrConfigs,
                                              TVisitor& visitor) {
            return VisitEqualUpToPriorFreqCtrs(indices, SingletonVector(ctrConfigs), visitor);
        };

        template <class TUint32>
        TWeightedBinFreqCalcer& ComputeFreq(const TCudaBuffer<TUint32, TMapping>& indices,
                                            const NCB::TCtrConfig& dstConfig,
                                            TCudaBuffer<float, TMapping>& dst) {
            TVisitor ctrVisitor = [&](const NCB::TCtrConfig& ctrConfig, const TCudaBuffer<float, TMapping>& ctr, ui32 stream) {
                Y_ASSERT(ctrConfig == dstConfig);
                Y_UNUSED(ctrConfig);
                dst.Reset(ctr.GetMapping());
                dst.Copy(ctr, stream);
            };
            VisitFreqCtrs(indices, dstConfig, ctrVisitor);
            return *this;
        };

    private:
        template <class T>
        using TBuffer = TCudaBuffer<T, TMapping>;

        TBuffer<const float> Weights;
        const float TotalWeight;

        TBuffer<float> BinWeights;
        TBuffer<float> Tmp;
        TBuffer<ui32> Bins;
        TBuffer<ui32> TempFlags;
        TBuffer<ui32> SegmentStarts;

        ui32 Mask;
        ui32 Stream;
    };

    extern template class THistoryBasedCtrCalcer<NCudaLib::TSingleMapping>;
    extern template class THistoryBasedCtrCalcer<NCudaLib::TMirrorMapping>;

    extern template class TWeightedBinFreqCalcer<NCudaLib::TSingleMapping>;
    extern template class TWeightedBinFreqCalcer<NCudaLib::TMirrorMapping>;
}
