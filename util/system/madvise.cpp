#include "madvise.h"
#include "align.h"
#include "info.h"

#include <util/generic/yexception.h>

#if defined(_win_)
#include <util/system/winint.h>
#else
#include <sys/types.h>
#include <sys/mman.h>
#endif

#ifndef MADV_DONTDUMP    /* This flag is defined in sys/mman.h since Linux 3.4, but currently old libc header is in use \
                            for capability with Ubuntu 12.04, so we need to define it here manually */
#define MADV_DONTDUMP 16 /* Explicity exclude from the core dump, overrides the coredump filter bits */
#endif

namespace {
    enum EMadvise {
        M_MADVISE_SEQUENTIAL = 0,
        M_MADVISE_RANDOM = 1,
        M_MADVISE_EVICT = 2,
        M_MADVISE_DONTDUMP = 3
    };

    void Madvise(EMadvise madv, const void* cbegin, size_t size) {
        static const size_t pageSize = NSystemInfo::GetPageSize();
        void* begin = AlignDown(const_cast<void*>(cbegin), pageSize);
        size = AlignUp(size, pageSize);

#if defined(_win_)
        if (M_MADVISE_EVICT == madv) {
            if (!VirtualFree((LPVOID)begin, size, MEM_DECOMMIT)) {
                TString err(LastSystemErrorText());
                ythrow yexception() << "VirtualFree(" << begin << ", " << size << ", " << MEM_DECOMMIT << ")"
                                    << " returned error: " << err;
            }
        }
#else
        static const int madviseFlags[] = {
            MADV_SEQUENTIAL,
            MADV_RANDOM,
#if defined(_linux_) || defined(_cygwin_)
            MADV_DONTNEED,
#else // freebsd, osx
            MADV_FREE,
#endif
            MADV_DONTDUMP
        };

        const int flag = madviseFlags[madv];

        if (-1 == madvise(begin, size, flag)) {
            TString err(LastSystemErrorText());
            ythrow yexception() << "madvise(" << begin << ", " << size << ", " << flag << ")"
                                << " returned error: " << err;
        }
#endif
    }
}

void MadviseSequentialAccess(const void* begin, size_t size) {
    Madvise(M_MADVISE_SEQUENTIAL, begin, size);
}

void MadviseRandomAccess(const void* begin, size_t size) {
    Madvise(M_MADVISE_RANDOM, begin, size);
}

void MadviseEvict(const void* begin, size_t size) {
    Madvise(M_MADVISE_EVICT, begin, size);
}

void MadviseExcludeFromCoreDump(const void* begin, size_t size) {
    Madvise(M_MADVISE_DONTDUMP, begin, size);
}
