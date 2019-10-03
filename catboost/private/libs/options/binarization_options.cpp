#include "binarization_options.h"
#include "option.h"
#include "restrictions.h"
#include "json_helper.h"

#include <library/json/json_value.h>

#include <util/digest/multi.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>
#include <util/string/split.h>


NCatboostOptions::TBinarizationOptions::TBinarizationOptions(
    const EBorderSelectionType borderSelectionType,
    const ui32 discretization,
    const ENanMode nanMode,
    ETaskType taskType)
    : BorderSelectionType("border_type", borderSelectionType)
    , BorderCount("border_count", discretization)
    , NanMode("nan_mode", nanMode)
    , TaskType(taskType)
{
}

void NCatboostOptions::TBinarizationOptions::DisableNanModeOption() {
    NanMode.SetDisabledFlag(true);
}

bool NCatboostOptions::TBinarizationOptions::operator==(const TBinarizationOptions& rhs) const {
    return std::tie(BorderSelectionType, BorderCount, NanMode) ==
        std::tie(rhs.BorderSelectionType, rhs.BorderCount, rhs.NanMode);
}

bool NCatboostOptions::TBinarizationOptions::operator!=(const TBinarizationOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TBinarizationOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options, &BorderSelectionType, &BorderCount, &NanMode);
    Validate();
}

void NCatboostOptions::TBinarizationOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, BorderSelectionType, BorderCount, NanMode);
}

void NCatboostOptions::TBinarizationOptions::Validate() const {
    CB_ENSURE(BorderCount.Get() <= GetMaxBinCount(TaskType), "Invalid border count: " << BorderCount.Get() << " (max border count: " << GetMaxBinCount(TaskType) << ")");
}

ui64 NCatboostOptions::TBinarizationOptions::GetHash() const {
    return MultiHash(BorderSelectionType, BorderCount, NanMode);
}

std::pair<TStringBuf, NJson::TJsonValue> NCatboostOptions::ParsePerFeatureBinarization(TStringBuf description) {
    std::pair<TStringBuf, NJson::TJsonValue> perFeatureBinarization;
    GetNext<TStringBuf>(description, ":", perFeatureBinarization.first);
    TBinarizationOptions binarizationOptions;

    for (const auto configItem : StringSplitter(description).Split(',').SkipEmpty()) {
        TStringBuf key, value;
        Split(configItem, '=', key, value);
        if (key == binarizationOptions.BorderCount.GetName()) {
            ui32 borderCount;
            CB_ENSURE(TryFromString(value, borderCount), "Couldn't parse border_count integer from string " << value);
            perFeatureBinarization.second[binarizationOptions.BorderCount.GetName()] = borderCount;
        } else if (key == binarizationOptions.BorderSelectionType.GetName()) {
            perFeatureBinarization.second[binarizationOptions.BorderSelectionType.GetName()] = value;
        } else if (key == binarizationOptions.NanMode.GetName()) {
            perFeatureBinarization.second[binarizationOptions.NanMode.GetName()] = value;
        } else {
            ythrow TCatBoostException() << "Unsupported float feature binarization option: " << key;
        }
    }
    return perFeatureBinarization;
}
