#pragma once

namespace NJson {
    class TJsonValue;
}

struct TCustomMetricDescriptor;
struct TCustomObjectiveDescriptor;
struct TCustomCallbackDescriptor;

void CheckFitParams(const NJson::TJsonValue& plainOptions,
                    const TCustomObjectiveDescriptor* objectiveDescriptor = nullptr,
                    const TCustomMetricDescriptor* evalMetricDescriptor = nullptr);
