#include "for_objects.h"

#include <library/cpp/testing/unittest/registar.h>


namespace NCB {
    namespace NDataNewUT {

    void Compare(const TQuantizedObjectsDataProvider& lhs, const TQuantizedObjectsDataProvider& rhs) {
#define COMPARE_OBJECTS_DATA_FIELD(field) \
        UNIT_ASSERT_EQUAL(lhs.Get##field(), rhs.Get##field());

        // Common Data
        COMPARE_OBJECTS_DATA_FIELD(ObjectCount)
        UNIT_ASSERT_EQUAL(*lhs.GetObjectsGrouping(), *rhs.GetObjectsGrouping());
        UNIT_ASSERT_EQUAL(*lhs.GetFeaturesLayout(), *rhs.GetFeaturesLayout());
        COMPARE_OBJECTS_DATA_FIELD(Order)
        COMPARE_OBJECTS_DATA_FIELD(GroupIds)
        COMPARE_OBJECTS_DATA_FIELD(SubgroupIds)
        COMPARE_OBJECTS_DATA_FIELD(Timestamp)

#undef COMPARE_OBJECTS_DATA_FIELD

        // Data
        UNIT_ASSERT_EQUAL(*lhs.GetQuantizedFeaturesInfo(), *rhs.GetQuantizedFeaturesInfo());

        NPar::TLocalExecutor localExecutor;

        lhs.GetFeaturesLayout()-> IterateOverAvailableFeatures<EFeatureType::Float>(
            [&] (TFloatFeatureIdx floatFeatureIdx) {
                UNIT_ASSERT_EQUAL(
                    (*lhs.GetFloatFeature(*floatFeatureIdx))->ExtractValues<ui16>(&localExecutor),
                    (*rhs.GetFloatFeature(*floatFeatureIdx))->ExtractValues<ui16>(&localExecutor)
                );
            }
        );

        lhs.GetFeaturesLayout()-> IterateOverAvailableFeatures<EFeatureType::Categorical>(
            [&] (TCatFeatureIdx catFeatureIdx) {
                UNIT_ASSERT_EQUAL(
                    (*lhs.GetCatFeature(*catFeatureIdx))->ExtractValues<ui32>(&localExecutor),
                    (*rhs.GetCatFeature(*catFeatureIdx))->ExtractValues<ui32>(&localExecutor)
                );
            }
        );
        lhs.GetFeaturesLayout()-> IterateOverAvailableFeatures<EFeatureType::Categorical>(
            [&] (TCatFeatureIdx catFeatureIdx) {
                UNIT_ASSERT_EQUAL(
                    lhs.GetCatFeatureUniqueValuesCounts(*catFeatureIdx),
                    rhs.GetCatFeatureUniqueValuesCounts(*catFeatureIdx)
                );
            }
        );
    }

    }

}
