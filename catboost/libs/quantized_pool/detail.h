#pragma once

#include <catboost/libs/column_description/column.h>

namespace NCB {
    namespace NQuantizationDetail {
        constexpr bool IsFloatColumn(const EColumn type) {
            return type == EColumn::Num ||
                type == EColumn::Label ||
                type == EColumn::Weight ||
                type == EColumn::GroupWeight;
        }

        constexpr bool IsDoubleColumn(const EColumn type) {
            return type == EColumn::Baseline;
        }

        constexpr bool IsUi32Column(const EColumn type) {
            return type == EColumn::SubgroupId ||
                type == EColumn::Categ;
        }

        constexpr bool IsUi64Column(const EColumn type) {
            return type == EColumn::SampleId ||
                type == EColumn::GroupId;
        }

        constexpr bool IsStringColumn(const EColumn type) {
            return type == EColumn::SampleId ||
                type == EColumn::GroupId ||
                type == EColumn::SubgroupId;
        }

        constexpr bool IsRequiredColumn(const EColumn type) {
            return IsFloatColumn(type) || IsDoubleColumn(type) ||
                IsUi32Column(type) || IsUi64Column(type);
        }

        constexpr ui32 GetFakeDocIdColumnIndex(ui32 columnCount) {
            return columnCount;
        }

        constexpr ui32 GetFakeGroupIdColumnIndex(ui32 columnCount) {
            return columnCount + 1;
        }

        constexpr ui32 GetFakeSubgroupIdColumnIndex(ui32 columnCount) {
            return columnCount + 2;
        }
    }
}
