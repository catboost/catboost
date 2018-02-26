#pragma once

#include "option.h"
#include "json_helper.h"
#include <library/grid_creator/binarization.h>
#include <util/digest/multi.h>

namespace NCatboostOptions {
    struct TBinarizationOptions {
        explicit TBinarizationOptions(const EBorderSelectionType borderSelectionType = EBorderSelectionType::GreedyLogSum,
                                      const ui32 discretization = 32,
                                      ENanMode nanMode = ENanMode::Forbidden)
            : BorderSelectionType("border_type", borderSelectionType)
            , BorderCount("border_count", discretization)
            , NanMode("nan_mode", nanMode)
        {
        }

        void DisableNanModeOption() {
            NanMode.SetDisabledFlag(true);
        }

        bool operator==(const TBinarizationOptions& rhs) const {
            return std::tie(BorderSelectionType, BorderCount, NanMode) ==
                   std::tie(rhs.BorderSelectionType, rhs.BorderCount, rhs.NanMode);
        }

        bool operator!=(const TBinarizationOptions& rhs) const {
            return !(rhs == *this);
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options, &BorderSelectionType, &BorderCount, &NanMode);
            Validate();
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options, BorderSelectionType, BorderCount, NanMode);
        }

        void Validate() const {
            CB_ENSURE(BorderCount.Get() < 256, "Invalid border count: " << BorderCount.Get());
        }

        ui64 GetHash() const {
            return MultiHash(BorderSelectionType, BorderCount, NanMode);
        }

        TOption<EBorderSelectionType> BorderSelectionType;
        TOption<ui32> BorderCount;
        TOption<ENanMode> NanMode;
    };
}

template <>
struct THash<NCatboostOptions::TBinarizationOptions> {
    size_t operator()(const NCatboostOptions::TBinarizationOptions& option) const noexcept {
        return option.GetHash();
    }
};
