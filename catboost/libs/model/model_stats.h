#pragma once

#include "model.h"

#include <catboost/libs/data/pool.h>

#include <util/generic/vector.h>

TVector<TVector<double>> ComputeTotalLeafWeights(const TPool& pool, const TFullModel& model);
