#pragma once

#include <util/system/defaults.h>

class TAhoCorasickCommon {
public:
    static ui32 GetVersion() {
        return 3;
    }

    static size_t GetBlockCount() {
        return 4;
    }
};
