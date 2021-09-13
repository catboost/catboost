#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/meta_info.h>
#include <catboost/libs/data/objects_grouping.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/data/quantized_features_info.h>
#include <catboost/libs/data/target.h>
#include <catboost/private/libs/data_types/groupid.h>
#include <catboost/libs/helpers/sparse_array.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/variant.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NCB {
    namespace NDataNewUT {

    template <class TValue>
    using TExpectedFeatureColumn = std::variant<TVector<TValue>, TConstPolymorphicValuesSparseArray<TValue, ui32>>;

    template <
        class TGroupIdData,
        class TSubgroupIdData,
        class TFloatFeature,
        class TCatFeature,
        class TTextFeature,
        class TEmbeddingFeature
    >
    struct TExpectedCommonObjectsData {
        EObjectsOrder Order = EObjectsOrder::Undefined;

        // Objects data
        TMaybe<TVector<TGroupIdData>> GroupIds;
        TMaybe<TVector<TSubgroupIdData>> SubgroupIds;
        TMaybe<TVector<ui64>> Timestamp;

        TVector<TMaybe<TExpectedFeatureColumn<TFloatFeature>>> FloatFeatures;
        TVector<TMaybe<TExpectedFeatureColumn<TCatFeature>>> CatFeatures;
        TVector<TMaybe<TExpectedFeatureColumn<TTextFeature>>> TextFeatures;
        TVector<TMaybe<TExpectedFeatureColumn<TEmbeddingFeature>>> EmbeddingFeatures;
    };

    /*
     *  GroupIds will be processed with CalcGroupIdFor
     *  SubgroupIds will be processed with CalcSubgroupIdFor
     *  CatFeatures will be processed with CalcCatFeatureHash
     */
    struct TExpectedRawObjectsData
        : public TExpectedCommonObjectsData<
              TStringBuf,
              TStringBuf,
              float,
              TStringBuf,
              TStringBuf,
              TVector<float>
          >
    {
        bool TreatGroupIdsAsIntegers = false;
    };

    // TODO(akhropov): quantized pools might have more complicated features data types in the future
    struct TExpectedQuantizedObjectsData
        : public TExpectedCommonObjectsData<TGroupId, TSubgroupId, ui8, ui32, TNothing, TNothing>
    {
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
        ui32 MaxCategoricalFeaturesUniqValuesOnLearn = 0;

        // needed for PackedBinaryFeaturesData and ExclusiveFeatureBundlesData
        THolder<TFeaturesArraySubsetIndexing> FullSubsetIndexing;

        TPackedBinaryFeaturesData PackedBinaryFeaturesData;
        TExclusiveFeatureBundlesData ExclusiveFeatureBundlesData;
        TFeatureGroupsData FeatureGroupsData;
        TMaybe<TVector<TCatFeatureUniqueValuesCounts>> CatFeatureUniqueValuesCounts;
    };


    template <class TExpectedObjectsData>
    struct TExpectedData {
        TDataMetaInfo MetaInfo;
        TExpectedObjectsData Objects;
        TObjectsGrouping ObjectsGrouping = TObjectsGrouping(0);
        TRawTargetData Target;
    };

    using TExpectedRawData = TExpectedData<TExpectedRawObjectsData>;

    using TExpectedQuantizedData = TExpectedData<TExpectedQuantizedObjectsData>;


    void CompareObjectsData(
        const TRawObjectsDataProvider& objectsData,
        const TExpectedRawData& expectedData,
        bool catFeaturesHashCanContainExtraData = false
    );

    void CompareObjectsData(
        const TQuantizedObjectsDataProvider& objectsData,
        const TExpectedQuantizedData& expectedData,
        bool catFeaturesHashCanContainExtraData = false
    );

    void CompareTargetData(
        const TRawTargetDataProvider& targetData,
        const TObjectsGrouping& expectedObjectsGrouping,
        const TRawTargetData& expectedData
    );

    template <class TObjectsDataProvider, class TExpectedObjectsDataProvider>
    void Compare(
        TDataProviderPtr dataProvider,
        const TExpectedData<TExpectedObjectsDataProvider>& expectedData,
        bool catFeaturesHashCanContainExtraData = false
    ) {
        const TIntrusivePtr<TDataProviderTemplate<TObjectsDataProvider>> subtypeDataProvider =
            dataProvider->CastMoveTo<TObjectsDataProvider>();

        UNIT_ASSERT(subtypeDataProvider);

        UNIT_ASSERT_EQUAL(subtypeDataProvider->MetaInfo, expectedData.MetaInfo);
            CompareObjectsData(
                *(subtypeDataProvider->ObjectsData),
                expectedData,
                catFeaturesHashCanContainExtraData
            );
        UNIT_ASSERT_EQUAL(*subtypeDataProvider->ObjectsGrouping, expectedData.ObjectsGrouping);
        CompareTargetData(
            subtypeDataProvider->RawTargetData,
            expectedData.ObjectsGrouping,
            expectedData.Target
        );
    }

    }
}
