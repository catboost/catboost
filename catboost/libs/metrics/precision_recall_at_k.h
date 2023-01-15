#pragma once

#include <util/generic/fwd.h>

double CalcPrecisionAtK(TConstArrayRef<double> approx, TConstArrayRef<float> target, int top, double border);

double CalcRecallAtK(TConstArrayRef<double> approx, TConstArrayRef<float> target, int top, double border);

double CalcAveragePrecisionK(TConstArrayRef<double> approx, TConstArrayRef<float> target, int top, double border);
