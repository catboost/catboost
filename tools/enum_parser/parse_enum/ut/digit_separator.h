#pragma once

enum class ELiterals {
    Char = sizeof(u8'.'),
    Int = 123'456'789,
    Float1 = int(456'789.123'456),
    Float2 = int(1'2e0'1),
    Float3 = int(0x1'2p4),
};

enum class ETimePrecision : unsigned long long {
    MicroSeconds    =               1   /* "us" */,
    MilliSeconds    =           1'000   /* "ms" */,
    Seconds         =       1'000'000   /* "s" */,
    Minutes         =      60'000'000   /* "m" */,
    Hours           =   3'600'000'000   /* "h" */,
    Days            =  86'400'000'000   /* "d" */,
    Weeks           = 604'800'000'000   /* "w" */,
};
