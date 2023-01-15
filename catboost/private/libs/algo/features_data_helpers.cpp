#include "features_data_helpers.h"


namespace NCB {

    THolder<IFeaturesBlockIterator> CreateFeaturesBlockIterator(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData,
        size_t start,
        size_t end) {

        Y_UNUSED(end);

        THashMap<ui32, ui32> columnReorderMap;
        CheckModelAndDatasetCompatibility(model, objectsData, &columnReorderMap);

        THolder<IFeaturesBlockIterator> featuresBlockIterator;

        if (const auto* rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(&objectsData)) {
            featuresBlockIterator = MakeHolder<TRawFeaturesBlockIterator>(
                model,
                *rawObjectsData,
                columnReorderMap,
                start);
        } else if (const auto* quantizedForCpuObjectsData
                       = dynamic_cast<const TQuantizedObjectsDataProvider*>(&objectsData))
        {
            /* TODO(akhropov): Implement block iterators for external columns and enable
             * base TQuantizedObjectsDataProvider support here.
             *  Not currently used as external columns are used only in GPU training
             */

            featuresBlockIterator = MakeHolder<TQuantizedFeaturesBlockIterator>(
                model,
                *quantizedForCpuObjectsData,
                columnReorderMap,
                start);
        } else {
            ythrow TCatBoostException() << "Unsupported objects data - neither raw nor quantized for CPU";
        }

        return featuresBlockIterator;
    }
}
