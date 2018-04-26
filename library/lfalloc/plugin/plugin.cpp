#include "plugin.h"
#include <library/lfalloc/lf_allocX64.h>

namespace NMalloc {

    static size_t Y_FORCE_INLINE Align(size_t value, size_t align) {
        return (value + align - 1) & ~(align - 1);
    }

    TAllocHeader* LFAlloc(size_t size, size_t signature) {
        size = Align(size, sizeof(TAllocHeader));
        TAllocHeader* header = (TAllocHeader*)::LFAlloc(size + sizeof(TAllocHeader));
        header->Encode(header, size, signature);
        return header;
    }

    void LFFree(void* p) {
        ::LFFree(p);
    }

    TAllocatorPlugin CreateLFPlugin(size_t signature) {
        TAllocatorPlugin result(signature, "lf", LFAlloc, LFFree);
        result.SetParam = LFAlloc_SetParam;
        result.GetParam = LFAlloc_GetParam;
        return result;
    };

}
