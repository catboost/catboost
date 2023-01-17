#include "blocked_histogram_helper.h"

void NCatboostCuda::TBlockedHistogramsHelper::Rebuild() {
    FeatureSlices.clear();
    BinFeatureSlices.clear();

    const ui32 MB = 1024 * 1024;
    //TODO(noxoomo): tune it +  specializations for 1Gbs, 10Gbs networks and infiniband
    const ui32 reduceBlockSize = NCudaLib::GetCudaManager().HasRemoteDevices() ? 256 * MB : 32 * MB;

    //TODO(noxoomo): there can be done more sophisticated balancing based on fold counts for feature
    //do it, if reduce'll be bottleneck in distributed setting
    const ui32 oneIntFeatureGroup = GetFeaturesPerInt(Policy);

    //depth is current depth, linear system is systems obtained after split
    const ui32 leavesCount = 1 << (Depth + 1);
    const ui32 singleLinearSystem = leavesCount + leavesCount * (leavesCount + 1) / 2;
    const ui32 meanFoldsPerFeature = MeanFoldCount();

    const ui32 groupSizeBytes = meanFoldsPerFeature * oneIntFeatureGroup * singleLinearSystem * sizeof(float);
    const ui64 maxGroupCount = ::NHelpers::CeilDivide(Grid.FeatureIds.size(), oneIntFeatureGroup);

    ui32 featuresPerGroup = Grid.FeatureIds.size();
    if (NCudaLib::GetCudaManager().GetDeviceCount() != 1) {
        const ui64 minGroupSize = ::NHelpers::CeilDivide(maxGroupCount, 2 * MaxStreamCount);
        featuresPerGroup = Max<ui32>(reduceBlockSize / groupSizeBytes, minGroupSize) * oneIntFeatureGroup;
    }

    ui32 binFeatureOffset = 0;
    NCudaLib::TDistributedObject<ui32> solutionOffsets = CreateDistributedObject<ui32>(0);

    for (ui32 firstFeatureInGroup = 0; firstFeatureInGroup < Grid.FeatureIds.size(); firstFeatureInGroup += featuresPerGroup) {
        const ui32 end = Min<ui32>(firstFeatureInGroup + featuresPerGroup, Grid.FeatureIds.size());
        FeatureSlices.push_back(TSlice(firstFeatureInGroup, end));
        ui32 binFeatureInSlice = 0;
        for (ui32 f = firstFeatureInGroup; f < end; ++f) {
            binFeatureInSlice += Grid.Folds[f];
        }

        BinFeatureSlices.push_back(TSlice(binFeatureOffset,
                                          binFeatureOffset + binFeatureInSlice));

        BeforeReduceMappings.push_back(NCudaLib::TStripeMapping::RepeatOnAllDevices(binFeatureInSlice));
        AfterReduceMappings.push_back(NCudaLib::TStripeMapping::SplitBetweenDevices(binFeatureInSlice));

        //how to get binFeatureIdx
        auto afterReduceLoadBinFeaturesSlice = CreateDistributedObject<TSlice>(TSlice(0, 0));
        auto blockSolutions = CreateDistributedObject<TSlice>(TSlice(0, 0));

        for (auto dev : AfterReduceMappings.back().NonEmptyDevices()) {
            auto afterReduceDeviceBinFeatures = AfterReduceMappings.back().DeviceSlice(dev);
            {
                TSlice loadSlice;
                loadSlice.Left = binFeatureOffset + afterReduceDeviceBinFeatures.Left;
                loadSlice.Right = binFeatureOffset + afterReduceDeviceBinFeatures.Right;
                afterReduceLoadBinFeaturesSlice.Set(dev, loadSlice);
            }

            {
                TSlice blockSolutionsOnDevice;

                blockSolutionsOnDevice.Left = solutionOffsets.At(dev);
                blockSolutionsOnDevice.Right = solutionOffsets.At(dev) + afterReduceDeviceBinFeatures.Size();
                blockSolutions.Set(dev, blockSolutionsOnDevice);
                solutionOffsets.Set(dev, solutionOffsets.At(dev) + afterReduceDeviceBinFeatures.Size());
            }
        }
        BlockedBinFeatures.push_back(afterReduceLoadBinFeaturesSlice);
        FlatResultsSlice.push_back(blockSolutions);

        binFeatureOffset += binFeatureInSlice;
    }

    NCudaLib::TMappingBuilder<NCudaLib::TStripeMapping> solutionsMappingBuilder;
    for (ui32 dev = 0; dev < NCudaLib::GetCudaManager().GetDeviceCount(); ++dev) {
        CB_ENSURE(
            solutionOffsets.At(dev) == FlatResultsSlice.back().At(dev).Right,
            "Solution offset and result slice boundary do not match at device " << dev);
        solutionsMappingBuilder.SetSizeAt(dev, solutionOffsets.At(dev));
    }

    FlatResultsMapping = solutionsMappingBuilder.Build();
}
