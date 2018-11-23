#pragma once

#include "columns.h"
#include "quantized_features_info.h"


namespace NCB {

    class TExternalFloatValuesHolder: public IQuantizedFloatValuesHolder {
    public:
        TExternalFloatValuesHolder(ui32 featureId,
                                   NCB::TMaybeOwningArrayHolder<float> srcData,
                                   const TFeaturesArraySubsetIndexing* subsetIndexing,
                                   TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : IQuantizedFloatValuesHolder(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SubsetIndexing(subsetIndexing)
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
        }

        THolder<IQuantizedFloatValuesHolder> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override;

        NCB::TMaybeOwningArrayHolder<ui8> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

    private:
        NCB::TMaybeOwningArrayHolder<float> SrcData;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;

        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };


    class TExternalCatValuesHolder: public IQuantizedCatValuesHolder {
    public:
        TExternalCatValuesHolder(ui32 featureId,
                                 NCB::TMaybeOwningArrayHolder<ui32> srcData,
                                 const TFeaturesArraySubsetIndexing* subsetIndexing,
                                 TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : IQuantizedCatValuesHolder(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SubsetIndexing(subsetIndexing)
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
        }

        THolder<IQuantizedCatValuesHolder> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override;

        NCB::TMaybeOwningArrayHolder<ui32> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

    private:
        NCB::TMaybeOwningArrayHolder<ui32> SrcData;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;

        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

}

