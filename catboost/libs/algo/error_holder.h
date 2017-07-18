#pragma once

struct TErrorHolder {
    double Error = 0;
    double Weight = 0;

    void Add(const TErrorHolder& other) {
        Error += other.Error;
        Weight += other.Weight;
    }
};

