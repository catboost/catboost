#pragma once

#include "columns.h"
#include "quantizations_manager.h"


namespace NCB {

    class TExternalFloatValuesHolder: public IQuantizedFloatValuesHolder {
    public:
        TExternalFloatValuesHolder(ui32 featureId,
                                   NCB::TMaybeOwningArrayHolder<float> srcData,
                                   const TFeaturesArraySubsetIndexing* subsetIndexing,
                                   TIntrusivePtr<TQuantizedFeaturesManager> featuresManager)
            : IQuantizedFloatValuesHolder(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SubsetIndexing(subsetIndexing)
            , FeaturesManager(std::move(featuresManager))
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
            FeatureManagerFeatureId = FeaturesManager->GetFeatureManagerId(*this);
        }

        NCB::TMaybeOwningArrayHolder<ui8> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

    private:
        NCB::TMaybeOwningArrayHolder<float> SrcData;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;

        TIntrusivePtr<TQuantizedFeaturesManager> FeaturesManager;
        ui32 FeatureManagerFeatureId = -1;
    };


    class TExternalCatValuesHolder: public IQuantizedCatValuesHolder {
    public:
        TExternalCatValuesHolder(ui32 featureId,
                                 NCB::TMaybeOwningArrayHolder<ui32> srcData,
                                 const TFeaturesArraySubsetIndexing* subsetIndexing,
                                 TIntrusivePtr<TQuantizedFeaturesManager> featuresManager)
            : IQuantizedCatValuesHolder(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SubsetIndexing(subsetIndexing)
            , FeaturesManager(std::move(featuresManager))
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
            FeatureManagerFeatureId = FeaturesManager->GetFeatureManagerId(*this);
        }

        NCB::TMaybeOwningArrayHolder<ui32> ExtractValues(NPar::TLocalExecutor* localExecutor) const override;

    private:
        NCB::TMaybeOwningArrayHolder<ui32> SrcData;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;

        TIntrusivePtr<TQuantizedFeaturesManager> FeaturesManager;
        ui32 FeatureManagerFeatureId = -1;
    };

}

