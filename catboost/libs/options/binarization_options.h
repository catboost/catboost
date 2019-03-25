#pragma once

#include "enums.h"
#include "option.h"

#include <library/grid_creator/binarization.h>

// TODO(yazevnul): make fwd header for library/json
namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    struct TBinarizationOptions {
        explicit TBinarizationOptions(
            EBorderSelectionType borderSelectionType = EBorderSelectionType::GreedyLogSum,
            ui32 discretization = 32,
            ENanMode nanMode = ENanMode::Forbidden,
            ETaskType taskType = ETaskType::CPU
        );

        void DisableNanModeOption();

        bool operator==(const TBinarizationOptions& rhs) const;
        bool operator!=(const TBinarizationOptions& rhs) const;

        void Load(const NJson::TJsonValue& options);

        void Save(NJson::TJsonValue* options) const;

        void Validate() const;

        ui64 GetHash() const;

        TOption<EBorderSelectionType> BorderSelectionType;
        TOption<ui32> BorderCount;
        TOption<ENanMode> NanMode;
    private:
        ETaskType TaskType;
    };
    std::pair<TStringBuf, NJson::TJsonValue> ParsePerFeatureBinarization(TStringBuf description);
}

template <>
struct THash<NCatboostOptions::TBinarizationOptions> {
    size_t operator()(const NCatboostOptions::TBinarizationOptions& option) const noexcept {
        return option.GetHash();
    }
};
