#pragma once

#include <string.h>
#include <util/system/compiler.h>

namespace NMalloc {
    struct TMallocInfo {
        TMallocInfo();

        const char* Name;

        bool (*SetParam)(const char* param, const char* value);
        const char* (*GetParam)(const char* param);

        bool (*CheckParam)(const char* param, bool defaultValue);
    };

    extern volatile bool IsAllocatorCorrupted;
    void AbortFromCorruptedAllocator();

    // this function should be implemented by malloc implementations
    TMallocInfo MallocInfo();

    struct TAllocHeader {
        void* Block;
        size_t AllocSize;
        void Y_FORCE_INLINE Encode(void* block, size_t size, size_t signature) {
            Block = block;
            AllocSize = size | signature;
        }
    };

    struct TAllocatorPlugin : public TMallocInfo {
        typedef TAllocHeader* (*TMallocFunction)(size_t size, size_t signature);
        typedef void (*TFreeFunction)(void*);
        TAllocatorPlugin(size_t signature, const char* name, TMallocFunction malloc, TFreeFunction free);
        TAllocatorPlugin() = default;

        Y_FORCE_INLINE size_t GetSignature() const {
            return Signature;
        }

        Y_FORCE_INLINE void* Allocate(size_t size) const {
            return MallocFunction(size, Signature) + 1;
        }

        Y_FORCE_INLINE void Free(void* ptr) const {
            FreeFunction(ptr);
        }

    private:
        TMallocFunction MallocFunction = nullptr;
        TFreeFunction FreeFunction = nullptr;
        size_t Signature = 0;
    };
}
