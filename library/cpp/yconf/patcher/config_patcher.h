#pragma once

#include <library/cpp/json/writer/json_value.h>

#include <util/generic/string.h>

namespace NConfigPatcher {
    struct TOptions {
        TString Prefix;
    };

    TString Patch(const TString& config, const TString& patch, const TOptions& options);
    TString Patch(const TString& config, const NJson::TJsonValue& parsedPatch, const TOptions& options);
    TString Diff(const TString& source, const TString& target);
}
