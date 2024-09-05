#include "madvise.h"
#include "align.h"
#include "info.h"

#include <util/generic/yexception.h>

#if defined(_win_)
    #include <util/system/winint.h>
#else
    #include <sys/mman.h>
#endif

#ifndef MADV_DONTDUMP        /* This flag is defined in sys/mman.h since Linux 3.4, but currently old libc header is in use \
                                for capability with Ubuntu 12.04, so we need to define it here manually */
    #define MADV_DONTDUMP 16 /* Explicity exclude from the core dump, overrides the coredump filter bits */
#endif

#ifndef MADV_DODUMP        /* This flag is defined in sys/mman.h since Linux 3.4, but currently old libc header is in use \
                                for capability with Ubuntu 12.04, so we need to define it here manually */
    #define MADV_DODUMP 17 /* Undo the effect of an earlier MADV_DONTDUMP */
#endif

namespace {
    void Madvise(int flag, const void* cbegin, size_t size) {
        static const size_t pageSize = NSystemInfo::GetPageSize();
        void* begin = AlignDown(const_cast<void*>(cbegin), pageSize);
        size = AlignUp(size, pageSize);

#if defined(_win_)
        if (!VirtualFree((LPVOID)begin, size, flag)) {
            TString err(LastSystemErrorText());
            ythrow yexception() << "VirtualFree(" << begin << ", " << size << ", " << flag << ")"
                                << " returned error: " << err;
        }
#else
        if (-1 == madvise(begin, size, flag)) {
            TString err(LastSystemErrorText());
            ythrow yexception() << "madvise(" << begin << ", " << size << ", " << flag << ")"
                                << " returned error: " << err;
        }
#endif
    }
} // namespace

void MadviseSequentialAccess(const void* begin, size_t size) {
#if !defined(_win_)
    Madvise(MADV_SEQUENTIAL, begin, size);
#endif
}

void MadviseSequentialAccess(TArrayRef<const char> data) {
    MadviseSequentialAccess(data.data(), data.size());
}

void MadviseSequentialAccess(TArrayRef<const ui8> data) {
    MadviseSequentialAccess(data.data(), data.size());
}

void MadviseRandomAccess(const void* begin, size_t size) {
#if !defined(_win_)
    Madvise(MADV_RANDOM, begin, size);
#endif
}

void MadviseRandomAccess(TArrayRef<const char> data) {
    MadviseRandomAccess(data.data(), data.size());
}

void MadviseRandomAccess(TArrayRef<const ui8> data) {
    MadviseRandomAccess(data.data(), data.size());
}

void MadviseEvict(const void* begin, size_t size) {
#if defined(_win_)
    Madvise(MEM_DECOMMIT, begin, size);
#elif defined(_linux_) || defined(_cygwin_)
    Madvise(MADV_DONTNEED, begin, size);
#else // freebsd, osx
    Madvise(MADV_FREE, begin, size);
#endif
}

void MadviseEvict(TArrayRef<const char> data) {
    MadviseEvict(data.data(), data.size());
}

void MadviseEvict(TArrayRef<const ui8> data) {
    MadviseEvict(data.data(), data.size());
}

void MadviseExcludeFromCoreDump(const void* begin, size_t size) {
#if defined(_darwin_)
    // Don't try to call function with flag which doesn't work
    // https://st.yandex-team.ru/PASSP-31755#6050bbafc68f501f2c22caab
    Y_UNUSED(begin);
    Y_UNUSED(size);
#elif !defined(_win_)
    Madvise(MADV_DONTDUMP, begin, size);
#endif
}

void MadviseExcludeFromCoreDump(TArrayRef<const char> data) {
    MadviseExcludeFromCoreDump(data.data(), data.size());
}

void MadviseExcludeFromCoreDump(TArrayRef<const ui8> data) {
    MadviseExcludeFromCoreDump(data.data(), data.size());
}

void MadviseIncludeIntoCoreDump(const void* begin, size_t size) {
#if defined(_darwin_)
    Y_UNUSED(begin);
    Y_UNUSED(size);
#elif !defined(_win_)
    Madvise(MADV_DODUMP, begin, size);
#endif
}

void MadviseIncludeIntoCoreDump(TArrayRef<const char> data) {
    MadviseIncludeIntoCoreDump(data.data(), data.size());
}

void MadviseIncludeIntoCoreDump(TArrayRef<const ui8> data) {
    MadviseIncludeIntoCoreDump(data.data(), data.size());
}
