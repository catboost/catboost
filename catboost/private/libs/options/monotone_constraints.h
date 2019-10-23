#pragma once

#include "load_options.h"

#include <library/json/writer/json_value.h>

#include <util/generic/fwd.h>
#include <util/generic/map.h>

TMap<TString, int> ParseMonotonicConstraintsFromString(const TString& monotoneConstraints);
void ConvertMonotoneConstraintsToCanonicalFormat(NJson::TJsonValue* catBoostJsonOptions);
