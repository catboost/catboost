#pragma once

#include "model.h"

#include <library/cpp/json/json_value.h>

NJson::TJsonValue GetPlainJsonWithAllOptions(const TFullModel& model);
