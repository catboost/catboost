#pragma once

#include <util/generic/fwd.h>

enum class EKappaMetricType;
struct TMetricHolder;

double CalcKappa(TMetricHolder confusionMatrix, int classCount, EKappaMetricType type);
