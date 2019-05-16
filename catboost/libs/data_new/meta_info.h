#pragma once

#include "features_layout.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/helpers/serialization.h>
#include <catboost/libs/model/features.h>

#include <library/binsaver/bin_saver.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/string/vector.h>
#include <util/system/types.h>


namespace NCB {

    struct TDataColumnsMetaInfo {
        TVector<TColumn> Columns;

    public:
        bool operator==(const TDataColumnsMetaInfo& rhs) const {
            return Columns == rhs.Columns;
        }

        SAVELOAD(Columns);

        ui32 CountColumns(const EColumn columnType) const;
        TVector<int> GetCategFeatures() const;
        void Validate() const;
        TVector<TString> GenerateFeatureIds(const TMaybe<TVector<TString>>& header) const;
    };

    struct TTargetStats {
        float MinValue = 0;
        float MaxValue = 0;
    };

    struct TDataMetaInfo {
        ui64 ObjectCount = 0;

        TFeaturesLayoutPtr FeaturesLayout;
        ui64 MaxCatFeaturesUniqValuesOnLearn = 0;

        bool HasTarget = false;
        TMaybe<TTargetStats> TargetStats;

        ui32 BaselineCount = 0;

        bool HasGroupId = false;
        bool HasGroupWeight = false;
        bool HasSubgroupIds = false;
        bool HasWeights = false;
        bool HasTimestamp = false;
        bool HasPairs = false;

        // can be set if baseline file header contains such information
        TVector<TString> ClassNames = {};

        // set only for dsv format pools
        // TODO(akhropov): temporary, serialization details shouldn't be here
        TMaybe<TDataColumnsMetaInfo> ColumnsInfo;

    public:
        TDataMetaInfo() = default;

        TDataMetaInfo(
            TMaybe<TDataColumnsMetaInfo>&& columnsInfo,
            bool hasAdditionalGroupWeight,
            bool hasPairs,
            TMaybe<ui32> additionalBaselineCount = Nothing(),

            // if specified - prefer these to Id in columnsInfo.Columns, otherwise take names
            TMaybe<const TVector<TString>*> featureNames = Nothing(),
            const TVector<TString>& classNames = {}
        );

        bool operator==(const TDataMetaInfo& rhs) const;

        void Validate() const;

        ui32 GetFeatureCount() const {
            return FeaturesLayout ? FeaturesLayout->GetExternalFeatureCount() : 0;
        }
    };

    void AddWithShared(IBinSaver* binSaver, TDataMetaInfo* data);
}
