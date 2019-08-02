#pragma once

#include "columns.h"
#include "quantized_features_info.h"


namespace NCB {

    class TExternalFloatValuesHolder: public ICloneableQuantizedFloatValuesHolder {
    public:
        TExternalFloatValuesHolder(ui32 featureId,
                                   NCB::TMaybeOwningConstArrayHolder<float> srcData,
                                   const TFeaturesArraySubsetIndexing* subsetIndexing,
                                   TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : ICloneableQuantizedFloatValuesHolder(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SubsetIndexing(subsetIndexing)
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
        }

        THolder<ICloneableQuantizedFloatValuesHolder> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override;

        NCB::TMaybeOwningArrayHolder<ui8> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

    private:
        NCB::TMaybeOwningConstArrayHolder<float> SrcData;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;

        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };


    class TExternalCatValuesHolder: public ICloneableQuantizedCatValuesHolder {
    public:
        TExternalCatValuesHolder(ui32 featureId,
                                 NCB::TMaybeOwningConstArrayHolder<ui32> srcData,
                                 const TFeaturesArraySubsetIndexing* subsetIndexing,
                                 TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : ICloneableQuantizedCatValuesHolder(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SubsetIndexing(subsetIndexing)
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
        }

        THolder<ICloneableQuantizedCatValuesHolder> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override;

        NCB::TMaybeOwningArrayHolder<ui32> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

    private:
        NCB::TMaybeOwningConstArrayHolder<ui32> SrcData;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;

        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

}

