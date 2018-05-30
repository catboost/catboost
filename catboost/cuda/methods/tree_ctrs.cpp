#include "tree_ctrs.h"

THolder<NCatboostCuda::TTreeCtrDataSetBuilder::TCompressedIndex>
NCatboostCuda::TTreeCtrDataSetBuilder::CreateCompressedIndex(NCudaLib::TSingleMapping docsMapping) {
    THolder<TCompressedIndex> dataSet = MakeHolder<TCompressedIndex>();
    const TVector<TCtr>& ctrs = TreeCtrDataSet.GetCtrs();

    const ui32 featureCount = static_cast<ui32>(ctrs.size());
    TVector<ui32> featureIds;
    featureIds.resize(featureCount);
    std::iota(featureIds.begin(), featureIds.end(), 0);

    TBinarizationInfoProvider binarizationInfoProvider(ctrs,
                                                       TreeCtrDataSet.FeaturesManager);
    TDataSetDescription description = {};
    description.Name = "TreeCtrs compressed dataset";

    using TBuilder = TSharedCompressedIndexBuilder<TSingleDevLayout>;
    ui32 id = TBuilder::AddDataSetToCompressedIndex(binarizationInfoProvider,
                                                    description,
                                                    docsMapping,
                                                    featureIds,
                                                    dataSet.Get());

    CB_ENSURE(id == 0);

    return dataSet;
}

NCatboostCuda::TTreeCtrDataSetBuilder::TConstVec
NCatboostCuda::TTreeCtrDataSetBuilder::GetBorders(const NCatboostCuda::TCtr& ctr,
                                                  const NCatboostCuda::TTreeCtrDataSetBuilder::TVec& floatCtr,
                                                  ui32 stream) {
    CB_ENSURE(TreeCtrDataSet.InverseCtrIndex.has(ctr));
    const ui32 featureId = TreeCtrDataSet.InverseCtrIndex[ctr];
    const auto& bordersSlice = TreeCtrDataSet.CtrBorderSlices[featureId];

    if (TreeCtrDataSet.AreCtrBordersComputed[featureId] == false) {
        const auto& binarizationDescription = TreeCtrDataSet.FeaturesManager.GetCtrBinarization(ctr);
        TCudaBuffer<float, NCudaLib::TSingleMapping> bordersVecSlice = TreeCtrDataSet.CtrBorders.SliceView(bordersSlice);
        ComputeCtrBorders(floatCtr,
                          binarizationDescription,
                          stream,
                          bordersVecSlice);
        TreeCtrDataSet.AreCtrBordersComputed[featureId] = true;
    }
    return TreeCtrDataSet.CtrBorders.SliceView(bordersSlice);
}

void NCatboostCuda::TTreeCtrDataSetBuilder::ComputeCtrBorders(const NCatboostCuda::TTreeCtrDataSetBuilder::TVec& ctr,
                                                              const NCatboostOptions::TBinarizationOptions& binarizationDescription,
                                                              ui32 stream,
                                                              NCatboostCuda::TTreeCtrDataSetBuilder::TVec& dst) {
    auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("Build ctr borders");
    CB_ENSURE(dst.GetMapping().GetObjectsSlice().Size() == binarizationDescription.BorderCount + 1);
    ComputeBordersOnDevice(ctr,
                           binarizationDescription,
                           dst,
                           stream);
}

void NCatboostCuda::TTreeCtrDataSetBuilder::operator()(const NCatboostCuda::TCtr& ctr,
                                                       const NCatboostCuda::TTreeCtrDataSetBuilder::TVec& floatCtr,
                                                       ui32 stream) {
    const ui32 featureId = TreeCtrDataSet.InverseCtrIndex[ctr];
    const auto borders = GetBorders(ctr, floatCtr, stream);
    auto profileGuard = NCudaLib::GetCudaManager().GetProfiler().Profile("binarizeOnDevice");
    BinarizeOnDevice(floatCtr,
                     borders,
                     TreeCtrDataSet.CompressedIndex->GetDataSet(0u).GetTCFeature(featureId),
                     TreeCtrDataSet.CompressedIndex->FlatStorage,
                     StreamParallelCtrVisits /* atomic update for 2 stream */,
                     IsIdentityPermutation ? nullptr : &GatherIndices,
                     stream);
}
