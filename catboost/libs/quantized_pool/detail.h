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
            return type == EColumn::SubgroupId;
        }

        constexpr bool IsUi64Column(const EColumn type) {
            return type == EColumn::DocId ||
                type == EColumn::GroupId;
        }

        constexpr bool IsRequiredColumn(const EColumn type) {
            return IsFloatColumn(type) || IsDoubleColumn(type) ||
                IsUi32Column(type) || IsUi64Column(type);
        }
    }
}
