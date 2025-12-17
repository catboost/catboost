#include "gpu_input_provider.h"

#include <catboost/libs/data/objects_grouping.h>

namespace NCB {

    TIntrusivePtr<TObjectsDataProvider> TGpuRawObjectsDataProvider::GetSubsetImpl(
        const TObjectsGroupingSubset& objectsGroupingSubset,
        TMaybe<TConstArrayRef<ui32>> ignoredFeatures,
        ui64 cpuRamLimit,
        NPar::ILocalExecutor* localExecutor
    ) const {
        Y_UNUSED(cpuRamLimit);

        TCommonObjectsData subsetCommonData = CommonData.GetSubset(objectsGroupingSubset, localExecutor);
        if (ignoredFeatures) {
            subsetCommonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(*subsetCommonData.FeaturesLayout);
            subsetCommonData.FeaturesLayout->IgnoreExternalFeatures(*ignoredFeatures);
        }

        return MakeIntrusive<TGpuRawObjectsDataProvider>(
            objectsGroupingSubset.GetSubsetGrouping(),
            std::move(subsetCommonData),
            TGpuInputData(Data),
            /*skipCheck*/ true
        );
    }

} // namespace NCB
