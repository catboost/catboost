#pragma once

#include <library/cpp/json/json_value.h>


class TFullModel;


NJson::TJsonValue GetPlainJsonWithAllOptions(const TFullModel& model);
