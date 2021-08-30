#include <stdlib.h>
#include <stdio.h>

#include "malloc.h"

namespace {
    bool SetEmptyParam(const char*, const char*) {
        return false;
    }

    const char* GetEmptyParam(const char*) {
        return nullptr;
    }

    bool CheckEmptyParam(const char*, bool defaultValue) {
        return defaultValue;
    }
}

namespace NMalloc {
    volatile bool IsAllocatorCorrupted = false;

    TMallocInfo::TMallocInfo()
        : Name()
        , SetParam(SetEmptyParam)
        , GetParam(GetEmptyParam)
        , CheckParam(CheckEmptyParam)
    {
    }

    void AbortFromCorruptedAllocator(const char* errorMessage) {
        errorMessage = errorMessage ? errorMessage : "<unspecified>";
        fprintf(stderr, "Allocator error: %s\n", errorMessage);
        IsAllocatorCorrupted = true;
        abort();
    }
}
