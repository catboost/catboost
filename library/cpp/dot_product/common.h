#pragma once

template <typename T>
struct TTriWayDotProduct {
    T LL = 1;
    T LR = 0;
    T RR = 1;
};

enum class ETriWayDotProductComputeMask: unsigned {
    // basic
    LL = 0b100,
    LR = 0b010,
    RR = 0b001,

    // useful combinations
    All = 0b111,
    Left = 0b110, // skip computation of R·R
    Right = 0b011, // skip computation of L·L
};
