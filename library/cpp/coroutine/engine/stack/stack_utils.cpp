#include "stack_utils.h"

#include <util/generic/scope.h>
#include <util/system/yassert.h>

#ifdef _linux_
#include <sys/mman.h>
#endif

#include <cerrno>
#include <cstdlib>
#include <cstring>


namespace NCoro::NStack {

#ifdef _linux_
    bool GetAlignedMemory(size_t sizeInPages, char*& rawPtr, char*& alignedPtr) noexcept {
        Y_ASSERT(sizeInPages);

        void* ptr = nullptr;
        int error = posix_memalign(&ptr, PageSize, sizeInPages * PageSize);
        alignedPtr = rawPtr = (char*)ptr;
        return rawPtr && alignedPtr && !error;
    }
#else
    bool GetAlignedMemory(size_t sizeInPages, char*& rawPtr, char*& alignedPtr) noexcept {
        Y_ASSERT(sizeInPages);

        rawPtr = (char*) malloc((sizeInPages + 1) * PageSize); // +1 in case result would be unaligned
        alignedPtr = (char*)( ((size_t)rawPtr + PageSize - 1) & ~PageSizeMask);
        return rawPtr && alignedPtr;
    }
#endif

#ifdef _linux_
    void ReleaseRss(char* alignedPtr, size_t numOfPages) noexcept {
        Y_ABORT_UNLESS( !((size_t)alignedPtr & PageSizeMask), "Not aligned pointer to release RSS memory");
        if (!numOfPages) {
            return;
        }
        if (auto res = madvise((void*) alignedPtr, numOfPages * PageSize, MADV_DONTNEED); res) {
            Y_ABORT_UNLESS(errno == EAGAIN || errno == ENOMEM, "Failed to release memory");
        }
    }
#else
    void ReleaseRss(char*, size_t) noexcept {
    }
#endif

#ifdef _linux_
    size_t CountMapped(char* alignedPtr, size_t numOfPages) noexcept {
        Y_ABORT_UNLESS( !((size_t)alignedPtr & PageSizeMask) );
        Y_ASSERT(numOfPages);

        size_t result = 0;
        unsigned char* mappedPages = (unsigned char*) calloc(numOfPages, numOfPages);
        Y_ABORT_UNLESS(mappedPages);
        Y_DEFER {
            free(mappedPages);
        };

        if (!mincore((void*)alignedPtr, numOfPages * PageSize, mappedPages)) {
            for (size_t i = 0; i < numOfPages; ++i) {
                if (mappedPages[i] & 1) {
                    ++result;
                }
            }
        } else {
            Y_ASSERT(false);
            return 0;
        }

        return result;
    }

#else
    size_t CountMapped(char*, size_t) noexcept {
        return 0; // stub for Windows tests
    }
#endif

}
