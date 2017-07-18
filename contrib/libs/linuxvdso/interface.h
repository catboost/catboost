#pragma once

#include <stddef.h>

namespace NVdso {
    struct TSymbol {
        inline TSymbol()
            : Name(0)
            , Address(0)
        {
        }

        inline TSymbol(const char* name, void* addr)
            : Name(name)
            , Address(addr)
        {
        }

        const char* Name;
        void* Address;
    };

    size_t Enumerate(TSymbol* s, size_t len);

    void* Function(const char* name, const char* version);
}
