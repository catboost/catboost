#pragma once

#include "load_options.h"

#include <library/cpp/json/writer/json_value.h>

#include <util/generic/fwd.h>
#include <util/generic/map.h>

void ConvertMonotoneConstraintsToCanonicalFormat(NJson::TJsonValue* catBoostJsonOptions);
