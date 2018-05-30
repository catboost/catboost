#include "tree_ctrs.h"

namespace NCatboostCuda {
    void TTreeCtrDataSet::BuildFeatureIndex() {
        CB_ENSURE(InverseCtrIndex.size() == 0, "Error: build could be done only once");

        for (const ui32 feature : CatFeatures) {
            TFeatureTensor tensor = BaseFeatureTensor;
            tensor.AddCatFeature(feature);
            const auto& configs = GetCtrsConfigsForTensor(tensor);
            for (auto& config : configs) {
                TCtr ctr;
                ctr.FeatureTensor = tensor;
                ctr.Configuration = config;
                const ui32 idx = static_cast<const ui32>(InverseCtrIndex.size());
                InverseCtrIndex[ctr] = idx;
                Ctrs.push_back(ctr);
                const auto borderCount = FeaturesManager.GetCtrBinarization(ctr).BorderCount;
                MaxBorderCount = Max<ui32>(MaxBorderCount, borderCount);
                const ui32 bordersSize = 1 + borderCount;
                const ui32 offset = static_cast<const ui32>(CtrBorderSlices.size() ? CtrBorderSlices.back().Right
                                                                                   : 0);
                const TSlice bordersSlice = TSlice(offset, offset + bordersSize);
                CtrBorderSlices.push_back(bordersSlice);
            }
        }

        TFeaturesMapping featuresMapping = CreateFeaturesMapping();

        auto bordersMapping = featuresMapping.Transform([&](TSlice deviceSlice) {
            ui32 size = 0;
            for (ui32 feature = static_cast<ui32>(deviceSlice.Left); feature < deviceSlice.Right; ++feature) {
                size += CtrBorderSlices[feature].Size();
            }
            return size;
        });
        CtrBorders.Reset(bordersMapping);

        if (CtrBorderSlices.size()) {
            //borders are so small, that it should be almost always faster to write all border vec then by parts
            TVector<float> borders(CtrBorderSlices.back().Right);
            bool needWrite = false;

            for (ui32 i = 0; i < Ctrs.size(); ++i) {
                const auto& ctr = Ctrs[i];
                AreCtrBordersComputed.push_back(false);
                if (FeaturesManager.IsKnown(ctr)) {
                    const auto& ctrBorders = FeaturesManager.GetBorders(FeaturesManager.GetId(ctr));
                    const ui64 offset = CtrBorderSlices[i].Left;
                    borders[offset] = ctrBorders.size();
                    std::copy(ctrBorders.begin(), ctrBorders.end(), borders.begin() + offset + 1);
                    CB_ENSURE(ctrBorders.size() < CtrBorderSlices[i].Size());
                    AreCtrBordersComputed.back() = true;
                    needWrite = true;
                }
            }
            if (needWrite) {
                CtrBorders.Write(borders);
            }
        }
    }
}
