#pragma once

struct TMetricHolder {
    double Error = 0;
    double Weight = 0;

    void Add(const TMetricHolder& other) {
        Error += other.Error;
        Weight += other.Weight;
    }
};

