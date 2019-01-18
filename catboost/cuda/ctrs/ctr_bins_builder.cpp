#include "ctr_bins_builder.h"

namespace NCatboostCuda {
    TCtrBinBuilder<NCudaLib::TSingleMapping> CreateBinBuilderForSingleDevice(const TCtrBinBuilder<NCudaLib::TMirrorMapping>& mirrorBuilder, ui32 deviceId,
                                                                             ui32 streamId) {
        TCtrBinBuilder<NCudaLib::TSingleMapping> singleDevBuilder(streamId);

        singleDevBuilder.LearnSlice = mirrorBuilder.LearnSlice;
        singleDevBuilder.TestSlice = mirrorBuilder.TestSlice;

        const auto singleDevMapping = mirrorBuilder.Indices.DeviceView(deviceId).GetMapping();

        singleDevBuilder.Indices.Reset(singleDevMapping);
        singleDevBuilder.Indices.Copy(mirrorBuilder.Indices.DeviceView(deviceId));

        singleDevBuilder.Bins.Reset(singleDevMapping);
        singleDevBuilder.Bins.Copy(mirrorBuilder.Bins.DeviceView(deviceId));

        singleDevBuilder.DecompressedTempBins.Reset(singleDevMapping);
        singleDevBuilder.Tmp.Reset(singleDevMapping);

        return singleDevBuilder;
    }

    template class TCtrBinBuilder<NCudaLib::TMirrorMapping>;
    template class TCtrBinBuilder<NCudaLib::TSingleMapping>;

}
