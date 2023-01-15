#pragma once

namespace NJson {
    class TJsonValue;
}

struct TCustomMetricDescriptor;
struct TCustomObjectiveDescriptor;

void CheckFitParams(const NJson::TJsonValue& plainOptions,
                    const TCustomObjectiveDescriptor* objectiveDescriptor = nullptr,
                    const TCustomMetricDescriptor* evalMetricDescriptor = nullptr);
