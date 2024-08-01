#pragma once

#include "features_layout.h"

#include <catboost/private/libs/options/enums.h>
#include <catboost/libs/column_description/column.h>
#include <catboost/libs/column_description/feature_tag.h>

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/dbg_output/dump.h>
#include <library/cpp/json/json_value.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

#include <limits>


namespace NCB {

    struct TDataColumnsMetaInfo {
        TVector<TColumn> Columns;

    public:
        bool operator==(const TDataColumnsMetaInfo& rhs) const {
            return Columns == rhs.Columns;
        }

        SAVELOAD(Columns);

        operator NJson::TJsonValue() const;

        ui32 CountColumns(const EColumn columnType) const;
        void Validate() const;
        TVector<TString> GenerateFeatureIds(const TMaybe<TVector<TString>>& header) const;
    };

    struct TTargetStats {
        float MinValue = std::numeric_limits<float>::max();
        float MaxValue = std::numeric_limits<float>::lowest();

    public:
        operator NJson::TJsonValue() const;

        void Update(float value) {
            if (value < MinValue) {
                MinValue = value;
            }
            if (value > MaxValue) {
                MaxValue = value;
            }
        }

        void Update(const TTargetStats& update) {
            if (update.MinValue < MinValue) {
                MinValue = update.MinValue;
            }
            if (update.MaxValue > MaxValue) {
                MaxValue = update.MaxValue;
            }
        }
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
        bool HasSampleId = false;
        bool HasWeights = false;
        bool HasTimestamp = false;
        bool HasPairs = false;
        bool HasGraph = false;
        bool StoreStringColumns = false;
        bool ForceUnitAutoPairWeights = false;

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
            bool hasGraph,
            bool loadSampleIds, // special flag because they are rarely used
            bool forceUnitAutoPairWeights,
            TMaybe<ui32> additionalBaselineCount = Nothing(),

            // if specified - prefer these to Id in columnsInfo.Columns, otherwise take names
            TMaybe<const TVector<TString>*> featureNames = Nothing(),
            TMaybe<const THashMap<TString, TTagDescription>*> featureTags = Nothing(),
            const TVector<NJson::TJsonValue>& classLabels = {}
        );

        bool EqualTo(const TDataMetaInfo& rhs, bool ignoreSparsity = false) const;

        bool operator==(const TDataMetaInfo& rhs) const {
            return EqualTo(rhs);
        }

        void Validate() const;

        operator NJson::TJsonValue() const;

        ui32 GetFeatureCount() const noexcept {
            return FeaturesLayout ? FeaturesLayout->GetExternalFeatureCount() : 0;
        }
    };

    void AddWithShared(IBinSaver* binSaver, TDataMetaInfo* data);
}

template <>
struct TDumper<NCB::TDataMetaInfo> {
    template <class S>
    static inline void Dump(S& s, const NCB::TDataMetaInfo& metaInfo) {
#define PRINT_META_INFO_FIELD(field) \
        s << #field"=" << metaInfo.field << Endl;

        PRINT_META_INFO_FIELD(ObjectCount);
        s << "FeaturesLayout=";
        if (metaInfo.FeaturesLayout) {
            s << DbgDump(metaInfo.FeaturesLayout);
        } else {
            s << "null";
        }
        s << Endl;
        PRINT_META_INFO_FIELD(MaxCatFeaturesUniqValuesOnLearn);
        PRINT_META_INFO_FIELD(TargetType);
        PRINT_META_INFO_FIELD(TargetCount);
        s << "TargetStats=";
        if (metaInfo.TargetStats.Defined()) {
            s << "{MinValue=" << metaInfo.TargetStats->MinValue
                << ",MaxValue=" << metaInfo.TargetStats->MaxValue << "}\n";
        } else {
            s << "undefined\n";
        }

        PRINT_META_INFO_FIELD(BaselineCount);
        PRINT_META_INFO_FIELD(HasGroupId);
        PRINT_META_INFO_FIELD(HasGroupWeight);
        PRINT_META_INFO_FIELD(HasSubgroupIds);
        PRINT_META_INFO_FIELD(HasWeights);
        PRINT_META_INFO_FIELD(HasTimestamp);
        PRINT_META_INFO_FIELD(HasPairs);
        PRINT_META_INFO_FIELD(HasGraph);

        s << "ClassLabels=" << DbgDump(metaInfo.ClassLabels) << Endl;

        s << "ColumnsInfo=";
        if (metaInfo.ColumnsInfo.Defined()) {
            s << "[\n";
            for (const auto& column : metaInfo.ColumnsInfo->Columns) {
                s << "\t{Type=" << column.Type << ",Id=" << column.Id << "}\n";
            }
            s << "]\n";
        } else {
            s << "undefined\n";
        }

#undef PRINT_META_INFO_FIELD
    }
};
