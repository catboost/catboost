#pragma once

#include <util/generic/vector.h>
#include <util/system/yassert.h>

// TODO(annaveronika): each metric should implement CreateMetricHolder(), CombineMetricHolders()
struct TMetricHolder {
    TVector<double> Stats = {0, 0};

    void Add(const TMetricHolder& other) {
        Y_ASSERT(Stats.size() == other.Stats.size());
        for (int i = 0; i < Stats.ysize(); ++i) {
            Stats[i] += other.Stats[i];
        }
    }
};

