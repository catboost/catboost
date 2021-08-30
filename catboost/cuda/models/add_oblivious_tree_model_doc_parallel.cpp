#include "add_oblivious_tree_model_doc_parallel.h"
#include "add_bin_values.h"
#include <catboost/cuda/cuda_lib/read_and_write_helpers.h>

namespace NCatboostCuda {
    namespace {
        class TComputeLeavesDocParallel {
        public:
            using TVec = TStripeBuffer<float>;
            using TDataSet = TDocParallelDataSet;
            using TCompressedIndex = typename TDataSet::TCompressedIndex;

            TComputeLeavesDocParallel(bool useStreams = false) {
                if (useStreams) {
                    const ui32 streamCount = 2;
                    for (ui32 i = 0; i < streamCount; ++i) {
                        Streams.push_back(NCudaLib::GetCudaManager().RequestStream());
                    }
                }
            }

            TComputeLeavesDocParallel& AddTask(const TObliviousTreeStructure& model,
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
                    Nodes.push_back(split.BinIdx);

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
                bins.Reset(NCudaLib::TMirrorMapping(Nodes.size()));
                ThroughHostBroadcast(Nodes, bins);

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
                                           features.AsConstBuf(),
                                           bins,
                                           cursor,
                                           stream);
            }

        private:
            TVector<TComputationStream> Streams;
            const TCompressedIndex* CompressedIndex = nullptr;

            TVector<TStripeBuffer<ui32>*> Cursors;
            TVector<TSlice> FeaturesSlices;
            TVector<ui8> Nodes;
            NCudaLib::TParallelStripeVectorBuilder<TCFeature> FeaturesBuilder;
        };

    }
    TAddModelDocParallel<TObliviousTreeModel>& TAddModelDocParallel<TObliviousTreeModel>::AddTask(const TObliviousTreeModel& model,
                                                                                                  const TAddModelDocParallel<TObliviousTreeModel>::TDataSet& dataSet,
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

    void TAddModelDocParallel<TObliviousTreeModel>::Proceed() {
        TMirrorBuffer<float> leaves;
        TMirrorBuffer<ui8> bins;
        TStripeBuffer<TCFeature> features;
        FeaturesBuilder.Build(features);
        leaves.Reset(NCudaLib::TMirrorMapping(CpuLeaves.size()));
        bins.Reset(NCudaLib::TMirrorMapping(FeatureBins.size()));
        ThroughHostBroadcast(CpuLeaves, leaves);
        ThroughHostBroadcast(FeatureBins, bins);

        if (!Streams.empty()) {
            NCudaLib::GetCudaManager().WaitComplete();
        }

        for (ui32 taskId = 0; taskId < Cursors.size(); ++taskId) {
            const auto& leavesSlice = LeavesSlices[taskId];
            const auto& featuresSlice = FeaturesSlices[taskId];

            auto taskValues = leaves.SliceView(leavesSlice);

            auto taskFeatures = NCudaLib::ParallelStripeView(features,
                                                             featuresSlice);

            auto taskBins = bins.SliceView(featuresSlice);

            const ui32 streamId = !Streams.empty() ? Streams[taskId % Streams.size()].GetId() : 0;
            Append(taskId,
                   taskFeatures,
                   taskBins,
                   taskValues,
                   streamId);
        }

        if (!Streams.empty()) {
            NCudaLib::GetCudaManager().WaitComplete();
        }
    }

    void TAddModelDocParallel<TObliviousTreeModel>::Append(ui32 taskId, const TStripeBuffer<TCFeature>& features,
                                                           const TMirrorBuffer<ui8>& bins, const TMirrorBuffer<float>& values,
                                                           ui32 stream) {
        auto& cursor = *Cursors[taskId];
        AddObliviousTree(CompressedIndex->GetStorage(), features.AsConstBuf(), bins, values, cursor, stream);
    }

    void ComputeBinsForModel(const TObliviousTreeStructure& structure, const TDocParallelDataSet& dataSet,
                             TStripeBuffer<ui32>* bins) {
        TComputeLeavesDocParallel computeLeaves;
        computeLeaves.AddTask(structure,
                              dataSet,
                              bins);
        computeLeaves.Proceed();
    }
}
