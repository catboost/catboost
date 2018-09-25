#pragma once

#include "features_layout.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/model/features.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/vector.h>
#include <util/string/vector.h>
#include <util/system/types.h>


namespace NCB {

    struct TDataColumnsMetaInfo {
        TVector<TColumn> Columns;

        ui32 CountColumns(const EColumn columnType) const;
        TVector<int> GetCategFeatures() const;
        void Validate() const;
        TVector<TString> GenerateFeatureIds(const TMaybe<TVector<TString>>& header) const;
    };

    struct TDataMetaInfo {
        TFeaturesLayout FeaturesLayout;

        bool HasTarget = false;

        ui32 BaselineCount = 0;

        bool HasGroupId = false;
        bool HasGroupWeight = false;
        bool HasSubgroupIds = false;
        bool HasWeights = false;
        bool HasTimestamp = false;
        bool HasPairs = false;

        // set only for dsv format pools
        // TODO(akhropov): temporary, serialization details shouldn't be here
        TMaybe<TDataColumnsMetaInfo> ColumnsInfo;

    public:
        TDataMetaInfo() = default;

        TDataMetaInfo(
            TVector<TColumn>&& columns,
            bool hasAdditionalGroupWeight,
            bool hasPairs,
            const TMaybe<TVector<TString>>& header = Nothing()
        );

        void Validate() const;

        ui32 GetFeatureCount() const {
            return (ui32)FeaturesLayout.GetExternalFeatureCount();
        }
    };

}
