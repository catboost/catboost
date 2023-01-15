#pragma once

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/vector.h>
#include <util/system/yassert.h>

// TODO(annaveronika): each metric should implement CreateMetricHolder(), CombineMetricHolders()
struct TMetricHolder {
    explicit TMetricHolder(int statsCount = 0) : Stats(statsCount) {}
    TVector<double> Stats;

    void Add(const TMetricHolder& other) {
        Y_VERIFY(Stats.empty() || other.Stats.empty() || Stats.size() == other.Stats.size());
        if (other.Stats.empty()) {
            return;
        }
        if (Stats.empty()) {
            Stats = other.Stats;
            return;
        }
        for (int i = 0; i < Stats.ysize(); ++i) {
            Stats[i] += other.Stats[i];
        }
    }
    SAVELOAD(Stats);
};

