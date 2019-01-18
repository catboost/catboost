#include "add_non_symmetric_tree_doc_parallel.h"
#include "add_bin_values.h"
#include <catboost/cuda/cuda_lib/read_and_write_helpers.h>
#include <catboost/cuda/models/kernel/add_model_value.cuh>

namespace NKernelHost {
    class TComputeNonSymmetricTreeBinsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCFeature> Features;
        TCudaBufferPtr<const TTreeNode> Nodes;
        TCudaBufferPtr<const ui32> DataSet;
        TCudaBufferPtr<ui32> Cursor;
        TCudaBufferPtr<const ui32> ReadIndices;
        TCudaBufferPtr<const ui32> WriteIndices;

    public:
        TComputeNonSymmetricTreeBinsKernel() = default;

        TComputeNonSymmetricTreeBinsKernel(
            TCudaBufferPtr<const TCFeature> features,
            TCudaBufferPtr<const TTreeNode> nodes,
            TCudaBufferPtr<const ui32> index,
            TCudaBufferPtr<ui32> cursor,
            TCudaBufferPtr<const ui32> readIndices = TCudaBufferPtr<const ui32>(),
            TCudaBufferPtr<const ui32> writeIndices = TCudaBufferPtr<const ui32>())
            : Features(features)
            , Nodes(nodes)
            , DataSet(index)
            , Cursor(cursor)
            , ReadIndices(readIndices)
            , WriteIndices(writeIndices)
        {
        }

        Y_SAVELOAD_DEFINE(Features, Nodes, DataSet, Cursor, ReadIndices, WriteIndices);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Cursor.Size() < (1ULL << 32));
            NKernel::ComputeNonSymmetricDecisionTreeBins(Features.Get(),
                                                         Nodes.Get(),
                                                         DataSet.Get(),
                                                         ReadIndices.Get(),
                                                         WriteIndices.Get(),
                                                         Cursor.Get(),
                                                         static_cast<ui32>(Cursor.Size()),
                                                         stream.GetStream());
        }
    };
}

namespace NCudaLib {
    REGISTER_KERNEL(0x1E7400, NKernelHost::TComputeNonSymmetricTreeBinsKernel);
}

namespace NCatboostCuda {
    namespace {
        class TComputeNonSymmetricTreeLeavesDocParallel {
        public:
            using TVec = TStripeBuffer<float>;
            using TDataSet = TDocParallelDataSet;
            using TCompressedIndex = typename TDataSet::TCompressedIndex;

            TComputeNonSymmetricTreeLeavesDocParallel(bool useStreams = false) {
                if (useStreams) {
                    const ui32 streamCount = 2;
                    for (ui32 i = 0; i < streamCount; ++i) {
                        Streams.push_back(NCudaLib::GetCudaManager().RequestStream());
                    }
                }
            }

            TComputeNonSymmetricTreeLeavesDocParallel& AddTask(const TNonSymmetricTreeStructure& model,
                                                               const TDataSet& dataSet,
                                                               TStripeBuffer<ui32>* bins) {
                if (CompressedIndex == nullptr) {
                    CompressedIndex = &dataSet.GetCompressedIndex();
                } else {
                    CB_ENSURE(CompressedIndex == &dataSet.GetCompressedIndex());
                }

                const TVector<TTreeNode>& treeNodes = model.GetNodes();

                TSlice featuresSlice = !FeaturesSlices.empty() ? FeaturesSlices.back() : TSlice(0, 0);
                featuresSlice.Left = featuresSlice.Right;
                featuresSlice.Right += treeNodes.size();
                FeaturesSlices.push_back(featuresSlice);

                for (ui32 i = 0; i < treeNodes.size(); ++i) {
                    const auto& node = treeNodes[i];
                    FeaturesBuilder.Add(dataSet.GetTCFeature((ui32)node.FeatureId));
                    Nodes.push_back(node);
                }
                Cursors.push_back(bins);

                return *this;
            }

            void Proceed() {
                TMirrorBuffer<TTreeNode> nodes;
                TStripeBuffer<TCFeature> features;
                FeaturesBuilder.Build(features);
                nodes.Reset(NCudaLib::TMirrorMapping(Nodes.size()));
                ThroughHostBroadcast(Nodes, nodes);

                TStripeBuffer<ui32> bins;

                if (Streams.size()) {
                    NCudaLib::GetCudaManager().WaitComplete();
                }

                for (ui32 taskId = 0; taskId < Cursors.size(); ++taskId) {
                    const auto& featuresSlice = FeaturesSlices[taskId];
                    auto taskFeatures = NCudaLib::ParallelStripeView(features,
                                                                     featuresSlice);

                    auto taskBins = nodes.SliceView(featuresSlice);

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
                         const TMirrorBuffer<TTreeNode>& nodes,
                         ui32 stream) {
                auto& cursor = *Cursors[taskId];
                using TKernel = NKernelHost::TComputeNonSymmetricTreeBinsKernel;
                LaunchKernels<TKernel>(cursor.NonEmptyDevices(),
                                       stream,
                                       features,
                                       nodes,
                                       CompressedIndex->GetStorage(),
                                       cursor);
            }

        private:
            TVector<TComputationStream> Streams;
            const TCompressedIndex* CompressedIndex = nullptr;

            TVector<TStripeBuffer<ui32>*> Cursors;
            TVector<TSlice> FeaturesSlices;
            TVector<TTreeNode> Nodes;
            NCudaLib::TParallelStripeVectorBuilder<TCFeature> FeaturesBuilder;
        };

    }

    TAddModelDocParallel<TNonSymmetricTree>& TAddModelDocParallel<TNonSymmetricTree>::AddTask(
        const TNonSymmetricTree& model,
        const TAddModelDocParallel<TNonSymmetricTree>::TDataSet& dataSet,
        TStripeBuffer<float>& cursor) {
        if (CompressedIndex == nullptr) {
            CompressedIndex = &dataSet.GetCompressedIndex();
        } else {
            CB_ENSURE(CompressedIndex == &dataSet.GetCompressedIndex());
        }
        const TVector<float>& modelValues = model.GetValues();
        Structures.push_back(model.GetStructure());
        DataSets.push_back(&dataSet);
        TSlice leavesSlice = LeavesSlices.size() ? LeavesSlices.back() : TSlice(0, 0);
        leavesSlice.Left = leavesSlice.Right;
        leavesSlice.Right += modelValues.size();
        LeavesSlices.push_back(leavesSlice);
        for (ui32 i = 0; i < modelValues.size(); ++i) {
            CpuLeaves.push_back(modelValues[i]);
        }
        Cursors.push_back(&cursor);
        TempBins.push_back(TStripeBuffer<ui32>::CopyMapping(cursor));

        return *this;
    }

    void TAddModelDocParallel<TNonSymmetricTree>::Proceed() {
        {
            TComputeNonSymmetricTreeLeavesDocParallel computeBins(Streams.size());
            for (ui64 i = 0; i < Structures.size(); ++i) {
                computeBins.AddTask(Structures[i], *DataSets[i], &TempBins[i]);
            }
            computeBins.Proceed();
        }
        TMirrorBuffer<float> leaves;
        leaves.Reset(NCudaLib::TMirrorMapping(CpuLeaves.size()));
        ThroughHostBroadcast(CpuLeaves, leaves);
        if (Streams.size()) {
            NCudaLib::GetCudaManager().WaitComplete();
        }
        for (ui32 taskId = 0; taskId < Cursors.size(); ++taskId) {
            const auto& leavesSlice = LeavesSlices[taskId];
            auto taskValues = leaves.SliceView(leavesSlice);

            const ui32 streamId = Streams.size() ? Streams[taskId % Streams.size()].GetId() : 0;
            AddBinModelValues(taskValues, TempBins[taskId], *Cursors[taskId], streamId);
        }
        if (Streams.size()) {
            NCudaLib::GetCudaManager().WaitComplete();
        }
    }

    void ComputeBinsForModel(const TNonSymmetricTreeStructure& structure,
                             const TDocParallelDataSet& dataSet,
                             TStripeBuffer<ui32>* bins) {
        TComputeNonSymmetricTreeLeavesDocParallel computeLeaves;
        computeLeaves.AddTask(structure,
                              dataSet,
                              bins);
        computeLeaves.Proceed();
    }
}
