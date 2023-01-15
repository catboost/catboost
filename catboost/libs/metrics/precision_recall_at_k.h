#pragma once

#include <util/generic/fwd.h>

double CalcPrecisionAtK(TConstArrayRef<double> approx, TConstArrayRef<float> target, int top, float border);

double CalcRecallAtK(TConstArrayRef<double> approx, TConstArrayRef<float> target, int top, float border);

double CalcAveragePrecisionK(TConstArrayRef<double> approx, TConstArrayRef<float> target, int top, float border);
