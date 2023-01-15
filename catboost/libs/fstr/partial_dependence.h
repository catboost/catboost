#pragma once

#include "util.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <library/cpp/threading/local_executor/local_executor.h>
#include <util/generic/vector.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/system/types.h>


TVector<double> GetPartialDependence(
        const TFullModel& model,
        const TVector<int>& features,
        const NCB::TDataProviderPtr dataProvider,
        int thread_count
);
