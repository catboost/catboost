#include "add_region_doc_parallel.h"
#include "add_bin_values.h"
#include <catboost/cuda/cuda_lib/read_and_write_helpers.h>
#include <catboost/cuda/models/kernel/add_model_value.cuh>

namespace NKernelHost {
    class TAddRegionKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCFeature> Features;
        TCudaBufferPtr<const TRegionDirection> Directions;
        TCudaBufferPtr<const float> Leaves;
        TCudaBufferPtr<const ui32> DataSet;
        TCudaBufferPtr<float> Cursor;
        TCudaBufferPtr<const ui32> ReadIndices;
        TCudaBufferPtr<const ui32> WriteIndices;

    public:
        TAddRegionKernel() = default;

        TAddRegionKernel(TCudaBufferPtr<const TCFeature> features,
                         TCudaBufferPtr<const TRegionDirection> bins,
                         TCudaBufferPtr<const float> leaves,
                         TCudaBufferPtr<const ui32> index,
                         TCudaBufferPtr<float> cursor,
                         TCudaBufferPtr<const ui32> readIndices = TCudaBufferPtr<const ui32>(),
                         TCudaBufferPtr<const ui32> writeIndices = TCudaBufferPtr<const ui32>())
            : Features(features)
            , Directions(bins)
            , Leaves(leaves)
            , DataSet(index)
            , Cursor(cursor)
            , ReadIndices(readIndices)
            , WriteIndices(writeIndices)
        {
        }

        Y_SAVELOAD_DEFINE(Features, Directions, Leaves, DataSet, Cursor, ReadIndices, WriteIndices);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Cursor.Size() < (1ULL << 32));
            CB_ENSURE(Directions.Size() == Features.Size());
            NKernel::AddRegion(Features.Get(), Directions.Get(), Leaves.Get(), (ui32)Directions.Size(), DataSet.Get(),
                               ReadIndices.Get(), WriteIndices.Get(), Cursor.Size(), Cursor.Get(),
                               Cursor.GetColumnCount(), Cursor.AlignedColumnSize(),
                               stream.GetStream());
        }
    };

    class TComputeRegionLeaveIndicesKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCFeature> Features;
        TCudaBufferPtr<const TRegionDirection> Splits;
        TCudaBufferPtr<const ui32> DataSet;
        TCudaBufferPtr<ui32> Cursor;
        TCudaBufferPtr<const ui32> ReadIndices;
        TCudaBufferPtr<const ui32> WriteIndices;

    public:
        TComputeRegionLeaveIndicesKernel() = default;

        TComputeRegionLeaveIndicesKernel(TCudaBufferPtr<const TCFeature> features,
                                         TCudaBufferPtr<const TRegionDirection> bins,
                                         TCudaBufferPtr<const ui32> index,
                                         TCudaBufferPtr<ui32> cursor,
                                         TCudaBufferPtr<const ui32> readIndices = TCudaBufferPtr<const ui32>(),
                                         TCudaBufferPtr<const ui32> writeIndices = TCudaBufferPtr<const ui32>())
            : Features(features)
            , Splits(bins)
            , DataSet(index)
            , Cursor(cursor)
            , ReadIndices(readIndices)
            , WriteIndices(writeIndices)
        {
        }

        Y_SAVELOAD_DEFINE(Features, Splits, DataSet, Cursor, ReadIndices, WriteIndices);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Cursor.Size() < (1ULL << 32));
            CB_ENSURE(Splits.Size() == Features.Size());
            NKernel::ComputeRegionBins(Features.Get(),
                                       Splits.Get(),
                                       (ui32)Splits.Size(),
                                       DataSet.Get(),
                                       ReadIndices.Get(),
                                       WriteIndices.Get(),
                                       Cursor.Get(),
                                       Cursor.Size(),
                                       stream.GetStream());
        }
    };
}

namespace NCudaLib {
    REGISTER_KERNEL(0x1E1400, NKernelHost::TAddRegionKernel);
    REGISTER_KERNEL(0x1E1401, NKernelHost::TComputeRegionLeaveIndicesKernel);
}

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

            TComputeLeavesDocParallel& AddTask(const TRegionStructure& model,
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
                    TRegionDirection direction;
                    direction.Bin = split.BinIdx;
                    direction.Value = GetSplitValue(model.Directions[i]);
                    Nodes.push_back(direction);

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
                TMirrorBuffer<TRegionDirection> directions;
                TStripeBuffer<TCFeature> features;
                FeaturesBuilder.Build(features);
                directions.Reset(NCudaLib::TMirrorMapping(Nodes.size()));
                ThroughHostBroadcast(Nodes, directions);

                if (Streams.size()) {
                    NCudaLib::GetCudaManager().WaitComplete();
                }

                for (ui32 taskId = 0; taskId < Cursors.size(); ++taskId) {
                    const auto& featuresSlice = FeaturesSlices[taskId];
                    auto taskFeatures = NCudaLib::ParallelStripeView(features,
                                                                     featuresSlice);

                    auto taskBins = directions.SliceView(featuresSlice);

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
                         const TMirrorBuffer<TRegionDirection>& directions,
                         ui32 stream) {
                auto& cursor = *Cursors[taskId];
                using TKernel = NKernelHost::TComputeRegionLeaveIndicesKernel;
                LaunchKernels<TKernel>(cursor.NonEmptyDevices(), stream, features, directions, CompressedIndex->GetStorage(), cursor);
            }

        private:
            TVector<TComputationStream> Streams;
            const TCompressedIndex* CompressedIndex = nullptr;

            TVector<TStripeBuffer<ui32>*> Cursors;
            TVector<TSlice> FeaturesSlices;
            TVector<TRegionDirection> Nodes;
            NCudaLib::TParallelStripeVectorBuilder<TCFeature> FeaturesBuilder;
        };

    }

    TAddModelDocParallel<TRegionModel>& TAddModelDocParallel<TRegionModel>::AddTask(const TRegionModel& model, const TAddModelDocParallel<TRegionModel>::TDataSet& dataSet, TStripeBuffer<float>& cursor) {
        if (CompressedIndex == nullptr) {
            CompressedIndex = &dataSet.GetCompressedIndex();
        } else {
            CB_ENSURE(CompressedIndex == &dataSet.GetCompressedIndex());
        }

        const TVector<float>& modelValues = model.GetValues();
        const TVector<TBinarySplit>& modelSplits = model.GetStructure().Splits;
        const auto& directions = model.GetStructure().Directions;
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
            TRegionDirection direction;
            direction.Bin = split.BinIdx;
            direction.Value = GetSplitValue(directions[i]);
            TreeNodes.push_back(direction);

            if (split.SplitType == EBinSplitType::TakeBin) {
                CB_ENSURE(dataSet.IsOneHot(split.FeatureId));
            } else {
                CB_ENSURE(!dataSet.IsOneHot(split.FeatureId));
            }
        }
        Cursors.push_back(&cursor);

        return *this;
    }

    void TAddModelDocParallel<TRegionModel>::Proceed() {
        TMirrorBuffer<float> leaves;
        TMirrorBuffer<TRegionDirection> directions;
        TStripeBuffer<TCFeature> features;
        FeaturesBuilder.Build(features);
        leaves.Reset(NCudaLib::TMirrorMapping(CpuLeaves.size()));
        directions.Reset(NCudaLib::TMirrorMapping(TreeNodes.size()));
        ThroughHostBroadcast(CpuLeaves, leaves);
        ThroughHostBroadcast(TreeNodes, directions);

        if (Streams.size()) {
            NCudaLib::GetCudaManager().WaitComplete();
        }

        for (ui32 taskId = 0; taskId < Cursors.size(); ++taskId) {
            const auto& leavesSlice = LeavesSlices[taskId];
            const auto& featuresSlice = FeaturesSlices[taskId];

            auto taskValues = leaves.SliceView(leavesSlice);

            auto taskFeatures = NCudaLib::ParallelStripeView(features,
                                                             featuresSlice);

            auto taskDirections = directions.SliceView(featuresSlice);

            const ui32 streamId = Streams.size() ? Streams[taskId % Streams.size()].GetId() : 0;
            Append(taskId,
                   taskFeatures,
                   taskDirections,
                   taskValues,
                   streamId);
        }

        if (Streams.size()) {
            NCudaLib::GetCudaManager().WaitComplete();
        }
    }

    void TAddModelDocParallel<TRegionModel>::Append(ui32 taskId,
                                                    const TStripeBuffer<TCFeature>& features,
                                                    const TMirrorBuffer<TRegionDirection>& directions,
                                                    const TMirrorBuffer<float>& values,
                                                    ui32 stream) {
        auto& cursor = *Cursors[taskId];
        using TKernel = NKernelHost::TAddRegionKernel;
        LaunchKernels<TKernel>(cursor.NonEmptyDevices(), stream, features, directions, values, CompressedIndex->GetStorage(), cursor);
    }

    void ComputeBinsForModel(const TRegionStructure& structure,
                             const TDocParallelDataSet& dataSet,
                             TStripeBuffer<ui32>* bins) {
        TComputeLeavesDocParallel computeLeaves;
        computeLeaves.AddTask(structure,
                              dataSet,
                              bins);
        computeLeaves.Proceed();
    }
}
