#pragma once

#include "add_bin_values.h"
#include "add_model.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/read_and_write_helpers.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>

namespace NCatboostCuda {
    template <>
    class TAddModelValue<TObliviousTreeModel, TDocParallelDataSet> {
    public:
        using TVec = TStripeBuffer<float>;
        using TDataSet = TDocParallelDataSet;
        using TCompressedIndex = typename TDataSet::TCompressedIndex;

        TAddModelValue(bool useStreams = false) {
            if (useStreams) {
                const ui32 streamCount = 2;
                for (ui32 i = 0; i < streamCount; ++i) {
                    Streams.push_back(NCudaLib::GetCudaManager().RequestStream());
                }
            }
        }

        TAddModelValue& AddTask(const TObliviousTreeModel& model,
                                const TDataSet& dataSet,
                                TStripeBuffer<float>& cursor) {
            if (CompressedIndex == nullptr) {
                CompressedIndex = &dataSet.GetCompressedIndex();
            } else {
                CB_ENSURE(CompressedIndex == &dataSet.GetCompressedIndex());
            }

            const TVector<float>& modelValues = model.GetValues();
            const TVector<TBinarySplit>& modelSplits = model.GetStructure().Splits;

            TSlice leavesSlice = LeavesSlices.size() ? LeavesSlices.back() : TSlice(0, 0);
            leavesSlice.Left = leavesSlice.Right;
            leavesSlice.Right += modelValues.size();
            LeavesSlices.push_back(leavesSlice);

            TSlice featuresSlice = FeaturesSlices.size() ? FeaturesSlices.back() : TSlice(0, 0);
            featuresSlice.Left = featuresSlice.Right;
            featuresSlice.Right += modelSplits.size();
            FeaturesSlices.push_back(featuresSlice);

            for (ui32 i = 0; i < modelValues.size(); ++i) {
                CpuLeaves.push_back(modelValues[i]);
            }

            for (ui32 i = 0; i < modelSplits.size(); ++i) {
                TBinarySplit split = modelSplits[i];
                FeaturesBuilder.Add(dataSet.GetTCFeature(split.FeatureId));
                FeatureBins.push_back(split.BinIdx);

                if (split.SplitType == EBinSplitType::TakeBin) {
                    CB_ENSURE(dataSet.IsOneHot(split.FeatureId));
                } else {
                    CB_ENSURE(!dataSet.IsOneHot(split.FeatureId));
                }
            }
            Cursors.push_back(&cursor);

            return *this;
        }

        void Proceed() {
            TMirrorBuffer<float> leaves;
            TMirrorBuffer<ui8> bins;
            TStripeBuffer<TCFeature> features;
            FeaturesBuilder.Build(features);
            leaves.Reset(NCudaLib::TMirrorMapping(CpuLeaves.size()));
            bins.Reset(NCudaLib::TMirrorMapping(FeatureBins.size()));
            ThroughHostBroadcast(CpuLeaves, leaves);
            ThroughHostBroadcast(FeatureBins, bins);

            if (Streams.size()) {
                NCudaLib::GetCudaManager().WaitComplete();
            }

            for (ui32 taskId = 0; taskId < Cursors.size(); ++taskId) {
                const auto& leavesSlice = LeavesSlices[taskId];
                const auto& featuresSlice = FeaturesSlices[taskId];

                auto taskValues = leaves.SliceView(leavesSlice);

                auto taskFeatures = NCudaLib::ParallelStripeView(features,
                                                                 featuresSlice);

                auto taskBins = bins.SliceView(featuresSlice);

                const ui32 streamId = Streams.size() ? Streams[taskId % Streams.size()].GetId() : 0;
                Append(taskId,
                       taskFeatures,
                       taskBins,
                       taskValues,
                       streamId);
            }

            if (Streams.size()) {
                NCudaLib::GetCudaManager().WaitComplete();
            }
        }

    private:
        void Append(ui32 taskId,
                    const TStripeBuffer<TCFeature>& features,
                    const TMirrorBuffer<ui8>& bins,
                    const TMirrorBuffer<float>& values,
                    ui32 stream) {
            auto& cursor = *Cursors[taskId];
            AddObliviousTree(CompressedIndex->GetStorage(), features, bins, values, cursor, stream);
        }

    private:
        TVector<TComputationStream> Streams;
        const TCompressedIndex* CompressedIndex = nullptr;

        TVector<TVec*> Cursors;

        TVector<TSlice> LeavesSlices;
        TVector<TSlice> FeaturesSlices;
        TVector<float> CpuLeaves;
        TVector<ui8> FeatureBins;
        NCudaLib::TStripeVectorBuilder<TCFeature> FeaturesBuilder;
    };

    template <>
    class TComputeLeaves<TObliviousTreeModel, TDocParallelDataSet> {
    public:
        using TVec = TStripeBuffer<float>;
        using TDataSet = TDocParallelDataSet;
        using TCompressedIndex = typename TDataSet::TCompressedIndex;

        TComputeLeaves(bool useStreams = false) {
            if (useStreams) {
                const ui32 streamCount = 2;
                for (ui32 i = 0; i < streamCount; ++i) {
                    Streams.push_back(NCudaLib::GetCudaManager().RequestStream());
                }
            }
        }

        TComputeLeaves& AddTask(const TObliviousTreeStructure& model,
                                const TDataSet& dataSet,
                                TStripeBuffer<ui32>* bins) {
            if (CompressedIndex == nullptr) {
                CompressedIndex = &dataSet.GetCompressedIndex();
            } else {
                CB_ENSURE(CompressedIndex == &dataSet.GetCompressedIndex());
            }

            const TVector<TBinarySplit>& modelSplits = model.Splits;

            TSlice featuresSlice = FeaturesSlices.size() ? FeaturesSlices.back() : TSlice(0, 0);
            featuresSlice.Left = featuresSlice.Right;
            featuresSlice.Right += modelSplits.size();
            FeaturesSlices.push_back(featuresSlice);

            for (ui32 i = 0; i < modelSplits.size(); ++i) {
                TBinarySplit split = modelSplits[i];
                FeaturesBuilder.Add(dataSet.GetTCFeature(split.FeatureId));
                FeatureBins.push_back(split.BinIdx);

                if (split.SplitType == EBinSplitType::TakeBin) {
                    CB_ENSURE(dataSet.IsOneHot(split.FeatureId));
                } else {
                    CB_ENSURE(!dataSet.IsOneHot(split.FeatureId));
                }
            }
            Cursors.push_back(bins);

            return *this;
        }

        void Proceed() {
            TMirrorBuffer<ui8> bins;
            TStripeBuffer<TCFeature> features;
            FeaturesBuilder.Build(features);
            bins.Reset(NCudaLib::TMirrorMapping(FeatureBins.size()));
            ThroughHostBroadcast(FeatureBins, bins);

            if (Streams.size()) {
                NCudaLib::GetCudaManager().WaitComplete();
            }

            for (ui32 taskId = 0; taskId < Cursors.size(); ++taskId) {
                const auto& featuresSlice = FeaturesSlices[taskId];
                auto taskFeatures = NCudaLib::ParallelStripeView(features,
                                                                 featuresSlice);

                auto taskBins = bins.SliceView(featuresSlice);

                const ui32 streamId = Streams.size() ? Streams[taskId % Streams.size()].GetId() : 0;
                Compute(taskId,
                        taskFeatures,
                        taskBins,
                        streamId);
            }

            if (Streams.size()) {
                NCudaLib::GetCudaManager().WaitComplete();
            }
        }

    private:
        void Compute(ui32 taskId,
                     const TStripeBuffer<TCFeature>& features,
                     const TMirrorBuffer<ui8>& bins,
                     ui32 stream) {
            auto& cursor = *Cursors[taskId];
            ComputeObliviousTreeLeaves(CompressedIndex->GetStorage(),
                                       features,
                                       bins,
                                       cursor,
                                       stream);
        }

    private:
        TVector<TComputationStream> Streams;
        const TCompressedIndex* CompressedIndex = nullptr;

        TVector<TStripeBuffer<ui32>*> Cursors;
        TVector<TSlice> FeaturesSlices;
        TVector<ui8> FeatureBins;
        NCudaLib::TStripeVectorBuilder<TCFeature> FeaturesBuilder;
    };

    inline void ComputeBinsForModel(const TObliviousTreeStructure& structure,
                                    const TDocParallelDataSet& dataSet,
                                    TStripeBuffer<ui32>* bins) {
        TComputeLeaves<TObliviousTreeModel, TDocParallelDataSet> computeLeaves;
        computeLeaves.AddTask(structure,
                              dataSet,
                              bins);
        computeLeaves.Proceed();
    }
}
