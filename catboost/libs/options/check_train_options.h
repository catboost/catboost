#pragma once

#include <library/json/json_reader.h>

struct TCustomMetricDescriptor;
struct TCustomObjectiveDescriptor;

void CheckFitParams(const NJson::TJsonValue& plainOptions,
                    const TCustomObjectiveDescriptor* objectiveDescriptor = nullptr,
                    const TCustomMetricDescriptor* evalMetricDescriptor = nullptr);
