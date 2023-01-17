#include "pairwise_score_calcer_for_policy.h"

NCatboostCuda::TComputePairwiseScoresHelper::TComputePairwiseScoresHelper(NCatboostCuda::EFeaturesGroupingPolicy policy,
                                                                          const NCatboostCuda::TComputePairwiseScoresHelper::TGpuDataSet& dataSet,
                                                                          const NCatboostCuda::TPairwiseOptimizationSubsets& subsets,
                                                                          TRandom& random,
                                                                          ui32 maxDepth,
                                                                          double l2Reg,
                                                                          double nonDiagReg,
                                                                          double rsm)
    : Policy(policy)
    , DataSet(dataSet)
    , Subsets(subsets)
    , MaxDepth(maxDepth)
    , NeedPointwiseWeights(subsets.GetPairwiseTarget().PointDer2OrWeights.GetObjectsSlice().Size() > 0)
    , LambdaDiag(l2Reg)
    , LambdaNonDiag(nonDiagReg)
{
    if (rsm < 1.0 && Policy != EFeaturesGroupingPolicy::BinaryFeatures) {
        SampleFeatures(random, rsm);
    }
    ResetHistograms();
}

NCatboostCuda::TComputePairwiseScoresHelper& NCatboostCuda::TComputePairwiseScoresHelper::Compute(TScopedCacheHolder& scoresCacheHolder,
                                                                                                  NCatboostCuda::TBinaryFeatureSplitResults* result) {
    if (GetCpuFeatureBuffer().GetObjectsSlice().Size() == 0) {
        return *this;
    }

    const auto& binaryFeatures = GetBinaryFeatures();
    const auto& gridCpu = GetCpuFeatureBuffer();
    const auto& gridGpu = GetGpuFeaturesBuffer();

    ++CurrentBit;
    Y_ASSERT(CurrentBit >= 0);
    if (static_cast<ui32>(CurrentBit) != Subsets.GetCurrentDepth() || CurrentBit == 0) {
        BuildFromScratch = true;
        CurrentBit = Subsets.GetCurrentDepth();
    }

    if (BuildFromScratch) {
        ResetHistograms();
    }

    auto& manager = NCudaLib::GetCudaManager();

    const ui32 rowSize = (1u << (CurrentBit + 1));
    const ui32 singleMatrixSize = rowSize * (rowSize + 1) / 2;
    const ui32 singleLinearSystemSize = singleMatrixSize + rowSize;

    auto& profiler = NCudaLib::GetProfiler();
    auto guard = profiler.Profile(TStringBuilder() << "Compute scores (" << Policy << ") for  #" << DataSet.GetGridSize(Policy)
                                                   << " features, depth " << CurrentBit);

    TBlockedHistogramsHelper blockedHelper(Policy,
                                           static_cast<ui32>(CurrentBit),
                                           GetCpuGrid(),
                                           MaxStreamCount);

    const ui32 blockCount = blockedHelper.GetBlockCount();
    const ui32 workingBlockCount = Min<ui32>(blockCount, MaxStreamCount);

    for (ui32 i = Streams.size(); i < workingBlockCount; ++i) {
        Streams.push_back(manager.RequestStream());
    }

    //it's temporary data, will be allocated for the biggest block
    //deallocation should be done after all computations to achieve compute/copy overlaps
    //linear systems are in ABCD ABCD ABCD ABCD layout
    //sqrt matrices are reduce-scattered data: A B C D
    //tempData will live while scoresCache is not destroyed
    auto& tempData = scoresCacheHolder.Cache(*this, 0, [&]() -> TTempData {
        TTempData data;
        data.LinearSystems.resize(workingBlockCount);
        data.SqrtMatrices.resize(workingBlockCount);
        {
            using TMappingBuilder = NCudaLib::TMappingBuilder<NCudaLib::TStripeMapping>;

            TVector<TMappingBuilder> linearSystemMapping(workingBlockCount);
            TVector<TMappingBuilder> sqrtMatricesMapping(workingBlockCount);

            for (ui32 blockId = 0; blockId < blockCount; ++blockId) {
                const ui32 workingBlockId = blockId % Streams.size();
                auto& linearSystemBlockMappingBuilder = linearSystemMapping[workingBlockId];
                auto& matrixBlockMappingBuilder = sqrtMatricesMapping[workingBlockId];

                auto reduceMapping = blockedHelper.ReduceMapping(blockId);

                for (auto dev : reduceMapping.NonEmptyDevices()) {
                    matrixBlockMappingBuilder.UpdateMaxSizeAt(dev, reduceMapping.DeviceSlice(dev).Size() *
                                                                       singleMatrixSize);
                }

                const ui32 linearSystemMemoryConsumption =
                    blockedHelper.GetBinFeatureCount(blockId) * singleMatrixSize;

                for (ui32 dev = 0; dev < NCudaLib::GetCudaManager().GetDeviceCount(); ++dev) {
                    linearSystemBlockMappingBuilder.UpdateMaxSizeAt(dev,
                                                                    linearSystemMemoryConsumption);
                }
            }

            for (ui32 workingBlock = 0; workingBlock < workingBlockCount; ++workingBlock) {
                data.LinearSystems[workingBlock].Reset(linearSystemMapping[workingBlock].Build());
                data.SqrtMatrices[workingBlock].Reset(sqrtMatricesMapping[workingBlock].Build());
            }
        }
        return data;
    });

    auto flatResultsMapping = blockedHelper.GetFlatResultsMapping();

    //vector equal to leaf count
    result->Solutions.Reset(flatResultsMapping.Transform([&](const TSlice slice) -> ui64 {
        return slice.Size();
    },
                                                         rowSize));
    //for weights in fstr
    result->MatrixDiagonal.Reset(flatResultsMapping.Transform([&](const TSlice slice) -> ui64 {
        return slice.Size();
    },
                                                              rowSize));
    //one per bin feature
    result->BinFeatures.Reset(flatResultsMapping.Transform([&](const TSlice slice) -> ui64 {
        return slice.Size();
    }));

    //one per score
    result->Scores.Reset(flatResultsMapping.Transform([&](const TSlice slice) -> ui64 {
        return slice.Size();
    }));

    if (result->LinearSystems) {
        result->LinearSystems->Reset(flatResultsMapping.Transform([&](const TSlice slice) -> ui64 {
            return slice.Size();
        },
                                                                  singleLinearSystemSize));
    }

    if (result->SqrtMatrices) {
        result->SqrtMatrices->Reset(flatResultsMapping.Transform([&](const TSlice slice) -> ui64 {
            return slice.Size();
        },
                                                                 singleMatrixSize));
    }

    for (ui32 blockId = 0; blockId < blockCount; ++blockId) {
        const int workingStreams = Min<int>(Streams.size(), blockCount);
        const ui32 groupId = blockId % Streams.size();
        const ui32 streamId = Streams[groupId].GetId();

        auto& linearSystem = tempData.LinearSystems[groupId];
        auto& sqrtMatrix = tempData.SqrtMatrices[groupId];
        auto blockGridCpu = blockedHelper.GetFeatures(gridCpu, blockId);
        auto blockGrid = blockedHelper.GetFeatures(gridGpu,
                                                   blockId);

        auto blockFoldsHist = blockedHelper.ComputeFoldsHistogram(blockId);
        auto blockBinFeaturesSlice = blockedHelper.GetBinFeatureSlice(blockId);

        const ui32 blockBinFeaturesCount = blockBinFeaturesSlice.Size();

        const auto& gatheredByLeavesTarget = Subsets.GetPairwiseTarget();

        if (NeedPointwiseWeights) {
            CB_ENSURE(gatheredByLeavesTarget.PointDer2OrWeights.GetObjectsSlice().Size(),
                     "No weights, use hist1 instead");

            ComputeBlockHistogram2(Policy,
                                   blockGrid,
                                   blockFoldsHist,
                                   blockBinFeaturesSlice,
                                   DataSet.GetCompressedIndex(),
                                   gatheredByLeavesTarget.PointWeightedDer,
                                   gatheredByLeavesTarget.PointDer2OrWeights,
                                   gatheredByLeavesTarget.Docs,
                                   Subsets.GetPointPartitions(),
                                   static_cast<ui32>(1 << CurrentBit),
                                   PointwiseHistograms,
                                   HistogramLineSize /* = total number of bin features */,
                                   BuildFromScratch,
                                   streamId);

        } else {
            CB_ENSURE(gatheredByLeavesTarget.PointDer2OrWeights.GetObjectsSlice().Size() == 0,
                     "There are weights, use hist2 instead");
            CB_ENSURE(
                PointwiseHistograms.GetMapping().SingleObjectSize() == 1,
                "Unexcepted object size " << PointwiseHistograms.GetMapping().SingleObjectSize());

            ComputeBlockHistogram1(Policy,
                                   blockGrid,
                                   blockBinFeaturesSlice,
                                   DataSet.GetCompressedIndex(),
                                   gatheredByLeavesTarget.PointWeightedDer,
                                   gatheredByLeavesTarget.Docs,
                                   Subsets.GetPointPartitions(),
                                   static_cast<ui32>(1 << CurrentBit),
                                   PointwiseHistograms,
                                   HistogramLineSize /* = total number of bin features */,
                                   BuildFromScratch,
                                   streamId);
        }

        {
            auto pairHistGuard = profiler.Profile(TStringBuilder() << "Pairwise hist (" << Policy << ") for  #" << blockGrid.GetObjectsSlice().Size()
                                                                   << " features, depth " << CurrentBit);

            ComputeBlockPairwiseHist2(Policy,
                                      blockGrid,
                                      blockGridCpu,
                                      blockFoldsHist,
                                      blockBinFeaturesSlice,
                                      DataSet.GetCompressedIndex(),
                                      gatheredByLeavesTarget.PairDer2OrWeights,
                                      gatheredByLeavesTarget.Pairs,
                                      Subsets.GetPairPartitions(),
                                      Subsets.GetPairPartStats(),
                                      CurrentBit,
                                      HistogramLineSize /* = total number of bin features */,
                                      BuildFromScratch,
                                      PairwiseHistograms,
                                      workingStreams,
                                      streamId);
        }

        //histograms are flat
        //linear system is for block
        //Reset will not reallocate memory as usual
        linearSystem.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(blockBinFeaturesCount,
                                                                        singleLinearSystemSize));
        FillBuffer(linearSystem, 0.0f, streamId);

        MakeLinearSystem(PointwiseHistograms,
                         Subsets.GetPointPartitionStats(),
                         PairwiseHistograms,
                         HistogramLineSize /* = total number of bin features */,
                         blockBinFeaturesSlice,
                         linearSystem,
                         streamId);

        {
            auto reducedMapping = blockedHelper.ReduceMapping(blockId);
            auto reducedLinearSystemsMapping = reducedMapping.Transform([&](const TSlice slice) -> ui64 {
                return slice.Size();
            },
                                                                        singleLinearSystemSize);

            ReduceScatter(linearSystem,
                          reducedLinearSystemsMapping,
                          IsReduceCompressed(),
                          streamId);
        }

        {
            auto sqrtMatrixMapping = blockedHelper.ReduceMapping(blockId).Transform([&](const TSlice slice) -> ui64 {
                return slice.Size();
            },
                                                                                    singleMatrixSize);
            sqrtMatrix.Reset(sqrtMatrixMapping);
        }

        auto flatResultsSlice = blockedHelper.GetFlatResultsSlice(blockId);

        //cholesky decomposition is done in-place on provided buffers + we need to store original system for score computation
        //linear system and sqrtMatrix are  zero-index
        //solutions are flat indexing for all blocks (so we could find best result in one kernel)

        PrepareSystemForCholesky(linearSystem,
                                 sqrtMatrix,
                                 flatResultsSlice,
                                 result->Solutions,
                                 result->MatrixDiagonal,
                                 streamId);

        Regularize(sqrtMatrix,
                   LambdaNonDiag, //bayesian -F weights adjust
                   LambdaDiag,    //classic l2 adjust
                   streamId);

        if (result->SqrtMatrices) {
            CopyReducedTempResult(sqrtMatrix.AsConstBuf(),
                                  flatResultsSlice,
                                  *result->SqrtMatrices,
                                  streamId);
        }

        //if only pairwise ders, then we don't need last row
        const bool removeLastRow = !NeedPointwiseWeights;

        //will mutate sqrtMatrix
        CholeskySolver(sqrtMatrix,
                       flatResultsSlice,
                       result->Solutions,
                       removeLastRow,
                       streamId);

        ComputeScores(binaryFeatures,
                      blockedHelper.GetBinFeatureAccessorSlice(blockId),
                      linearSystem,
                      flatResultsSlice,
                      result->Solutions,
                      result->BinFeatures,
                      result->Scores,
                      streamId);

        if (result->LinearSystems) {
            CopyReducedTempResult(linearSystem.AsConstBuf(),
                                  flatResultsSlice,
                                  *result->LinearSystems,
                                  streamId);
        }
    }
    Synchronized = false;
    BuildFromScratch = false;
    return *this;
}

void NCatboostCuda::TComputePairwiseScoresHelper::ResetHistograms() {
    const ui32 binFeatureCount = static_cast<const ui32>(GetBinaryFeatures().GetObjectsSlice().Size());
    HistogramLineSize = binFeatureCount;

    const ui32 maxParts = (1 << (MaxDepth - 1));
    auto pairwiseHistMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(binFeatureCount * maxParts * maxParts, 4);
    auto pointwiseHistMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(maxParts * binFeatureCount, NeedPointwiseWeights ? 2 : 1);

    PairwiseHistograms.Reset(pairwiseHistMapping);
    PointwiseHistograms.Reset(pointwiseHistMapping);

    FillBuffer(PairwiseHistograms, 0.0f);
    FillBuffer(PointwiseHistograms, 0.0f);

    NCudaLib::GetCudaManager().DefaultStream().Synchronize();
}

const TStripeBuffer<TCFeature>& NCatboostCuda::TComputePairwiseScoresHelper::GetGpuFeaturesBuffer() const {
    if (IsSampledGrid) {
        ValidateSampledGrid();
        return *GpuGrid;
    } else {
        return DataSet.GetGrid(Policy);
    }
}

const NCatboostCuda::TCpuGrid& NCatboostCuda::TComputePairwiseScoresHelper::GetCpuGrid() const {
    if (IsSampledGrid) {
        ValidateSampledGrid();
        return *CpuGrid;
    } else {
        return DataSet.GetCpuGrid(Policy);
    }
}

TMirrorBuffer<const TCBinFeature>& NCatboostCuda::TComputePairwiseScoresHelper::GetBinaryFeatures() const {
    if (IsSampledGrid) {
        ValidateSampledGrid();

        return CacheHolder.Cache(DataSet, Policy, [&]() -> TMirrorBuffer<const TCBinFeature> {
            TMirrorBuffer<TCBinFeature> mirrorBinFeatures;
            mirrorBinFeatures.Reset(NCudaLib::TMirrorMapping(BinFeaturesCpu->size()));
            mirrorBinFeatures.Write(*BinFeaturesCpu);
            NCudaLib::GetCudaManager().Barrier();
            return mirrorBinFeatures.ConstCopyView();
        });

    } else {
        auto& cachedBinFeatures = DataSet.GetCacheHolder().Cache(DataSet, Policy, [&]() -> TMirrorBuffer<const TCBinFeature> {
            TMirrorBuffer<TCBinFeature> mirrorBinFeatures;
            mirrorBinFeatures.Reset(NCudaLib::TMirrorMapping(DataSet.GetBinFeatures(Policy).size()));
            mirrorBinFeatures.Write(DataSet.GetBinFeatures(Policy));
            NCudaLib::GetCudaManager().Barrier();
            return mirrorBinFeatures.ConstCopyView();
        });

        return cachedBinFeatures;
    }
}

TCudaBuffer<const TCFeature, NCatboostCuda::TComputePairwiseScoresHelper::TFeaturesMapping, NCudaLib::EPtrType::CudaHost>& NCatboostCuda::TComputePairwiseScoresHelper::GetCpuFeatureBuffer() const {
    auto computeFunc = [&]() -> TCudaBuffer<const TCFeature, TFeaturesMapping, NCudaLib::EPtrType::CudaHost> {
        const auto& grid = GetGpuFeaturesBuffer();
        auto features = TCudaBuffer<TCFeature, NCudaLib::TStripeMapping, NCudaLib::EPtrType::CudaHost>::CopyMapping(grid);
        features.Copy(grid);
        NCudaLib::GetCudaManager().Barrier();
        return features.ConstCopyView();
    };

    if (IsSampledGrid) {
        ValidateSampledGrid();
        return CacheHolder.Cache(DataSet, Policy, computeFunc);
    } else {
        return DataSet.GetCacheHolder().Cache(DataSet, Policy, computeFunc);
    }
}

void NCatboostCuda::TComputePairwiseScoresHelper::SampleFeatures(TRandom& random, double rsm) {
    const TCpuGrid& srcGrid = DataSet.GetCpuGrid(Policy);
    const auto& featureIds = srcGrid.FeatureIds;
    CB_ENSURE(featureIds.size());
    CB_ENSURE(rsm > 1e-2, "Too low rsm " << rsm);

    const ui32 featuresPerInt = GetFeaturesPerInt(Policy);

    {
        TVector<ui32> sampledFeatures;

        for (ui32 firstFeature = 0; firstFeature < featureIds.size(); firstFeature += featuresPerInt) {
            if (random.NextUniform() > rsm) {
                continue;
            }
            const ui32 lastFeature = Min<ui32>(firstFeature + featuresPerInt, static_cast<const ui32>(featureIds.size()));
            for (ui32 i = firstFeature; i < lastFeature; ++i) {
                sampledFeatures.push_back(i);
            }
        }

        if (sampledFeatures.size() == 0) {
            double nextRsm = Min<double>(rsm * 2.0, 1.0);
            return SampleFeatures(random, nextRsm);
        }

        if (sampledFeatures.size() == featureIds.size()) {
            //no need to sampling
            return;
        }
        CpuGrid = srcGrid.Subgrid(sampledFeatures);
    }
    CATBOOST_DEBUG_LOG << "Sample features for policy " << Policy << " #" << CpuGrid->FeatureIds.size() << Endl;

    BinFeaturesCpu = TVector<TCBinFeature>();
    NCudaLib::TParallelStripeVectorBuilder<TCFeature> featuresBuilder;

    ui32 firstFoldIdx = 0;
    for (ui32 i = 0; i < CpuGrid->FeatureIds.size(); ++i) {
        const ui32 featureId = CpuGrid->FeatureIds[i];
        ui32 folds = CpuGrid->Folds[i];
        for (ui32 fold = 0; fold < folds; ++fold) {
            TCBinFeature binFeature;
            binFeature.FeatureId = featureId;
            binFeature.BinId = fold;
            BinFeaturesCpu->push_back(binFeature);
        }
        auto feature = DataSet.GetTCFeature(featureId);
        for (ui32 dev = 0; dev < feature.DeviceCount(); ++dev) {
            feature[dev].FirstFoldIndex = firstFoldIdx;
        }
        firstFoldIdx += folds;
        featuresBuilder.Add(feature);
    }
    GpuGrid = TStripeBuffer<TCFeature>();
    featuresBuilder.Build(*GpuGrid);
    IsSampledGrid = true;
}

void NCatboostCuda::TComputePairwiseScoresHelper::ValidateSampledGrid() const {
    CB_ENSURE(GpuGrid.Defined());
    CB_ENSURE(CpuGrid.Defined());
    CB_ENSURE(BinFeaturesCpu.Defined());
}
