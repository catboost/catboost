#pragma once

#include "features_layout.h"

#include <catboost/private/libs/options/enums.h>
#include <catboost/libs/column_description/column.h>

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/json/json_value.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
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

        ERawTargetType TargetType = ERawTargetType::None;
        ui32 TargetCount = 0;
        TMaybe<TTargetStats> TargetStats;

        ui32 BaselineCount = 0;

        bool HasGroupId = false;
        bool HasGroupWeight = false;
        bool HasSubgroupIds = false;
        bool HasWeights = false;
        bool HasTimestamp = false;
        bool HasPairs = false;

        // can be set from baseline file header or from quantized pool
        TVector<NJson::TJsonValue> ClassLabels = {};

        // set only for dsv format pools
        // TODO(akhropov): temporary, serialization details shouldn't be here
        TMaybe<TDataColumnsMetaInfo> ColumnsInfo;

    public:
        TDataMetaInfo() = default;

        TDataMetaInfo(
            TMaybe<TDataColumnsMetaInfo>&& columnsInfo,
            ERawTargetType targetType,
            bool hasAdditionalGroupWeight,
            bool hasTimestamp,
            bool hasPairs,
            TMaybe<ui32> additionalBaselineCount = Nothing(),

            // if specified - prefer these to Id in columnsInfo.Columns, otherwise take names
            TMaybe<const TVector<TString>*> featureNames = Nothing(),
            const TVector<NJson::TJsonValue>& classLabels = {}
        );

        bool EqualTo(const TDataMetaInfo& rhs, bool ignoreSparsity = false) const;

        bool operator==(const TDataMetaInfo& rhs) const {
            return EqualTo(rhs);
        }

        void Validate() const;

        ui32 GetFeatureCount() const {
            return FeaturesLayout ? FeaturesLayout->GetExternalFeatureCount() : 0;
        }
    };

    void AddWithShared(IBinSaver* binSaver, TDataMetaInfo* data);
}
