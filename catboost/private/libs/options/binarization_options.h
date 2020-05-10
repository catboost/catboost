#pragma once

#include "enums.h"
#include "option.h"

#include <library/cpp/grid_creator/binarization.h>

// TODO(yazevnul): make fwd header for library/cpp/json
namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    struct TBinarizationOptions {
        explicit TBinarizationOptions(
            EBorderSelectionType borderSelectionType = EBorderSelectionType::GreedyLogSum,
            ui32 discretization = 32,
            ENanMode nanMode = ENanMode::Forbidden,
            ui32 maxSubsetSize = 200000
        );

        void DisableNanModeOption();
        void DisableMaxSubsetSizeForBuildBordersOption();

        bool operator==(const TBinarizationOptions& rhs) const;
        bool operator!=(const TBinarizationOptions& rhs) const;

        void Load(const NJson::TJsonValue& options);

        void Save(NJson::TJsonValue* options) const;

        void Validate() const;

        ui64 GetHash() const;

        TOption<EBorderSelectionType> BorderSelectionType;
        TOption<ui32> BorderCount;
        TOption<ENanMode> NanMode;
        TOption<ui32> MaxSubsetSizeForBuildBorders;
    };
    std::pair<TStringBuf, NJson::TJsonValue> ParsePerFeatureBinarization(TStringBuf description);
}

template <>
struct THash<NCatboostOptions::TBinarizationOptions> {
    size_t operator()(const NCatboostOptions::TBinarizationOptions& option) const noexcept {
        return option.GetHash();
    }
};
