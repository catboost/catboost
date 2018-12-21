#include "binarization_options.h"
#include "option.h"
#include "json_helper.h"

#include <library/json/json_value.h>

#include <util/digest/multi.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>


NCatboostOptions::TBinarizationOptions::TBinarizationOptions(
    const EBorderSelectionType borderSelectionType,
    const ui32 discretization,
    const ENanMode nanMode)
    : BorderSelectionType("border_type", borderSelectionType)
    , BorderCount("border_count", discretization)
    , NanMode("nan_mode", nanMode)
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
    CB_ENSURE(BorderCount.Get() < 256, "Invalid border count: " << BorderCount.Get());
}

ui64 NCatboostOptions::TBinarizationOptions::GetHash() const {
    return MultiHash(BorderSelectionType, BorderCount, NanMode);
}
