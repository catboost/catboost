#include <util/generic/yexception.h>
#include "align.h"
#include "error.h"
#include "info.h"
#include "mlock.h"

#if defined(_unix_)
#include <sys/mman.h>
#if !defined(MCL_ONFAULT) && defined(MCL_FUTURE) // Old glibc.
#define MCL_ONFAULT (MCL_FUTURE << 1)
#endif
#if defined(_android_)
#include <sys/syscall.h>
#define munlockall() syscall(__NR_munlockall)
#endif
#else
#include "winint.h"
#endif

void LockMemory(const void* addr, size_t len) {
#if defined(_unix_)
    const size_t pageSize = NSystemInfo::GetPageSize();
    if (mlock(AlignDown(addr, pageSize), AlignUp(len, pageSize)))
        ythrow yexception() << LastSystemErrorText();
#elif defined(_win_)
    HANDLE hndl = GetCurrentProcess();
    SIZE_T min, max;
    if (!GetProcessWorkingSetSize(hndl, &min, &max))
        ythrow yexception() << LastSystemErrorText();
    if (!SetProcessWorkingSetSize(hndl, min + len, max + len))
        ythrow yexception() << LastSystemErrorText();
    if (!VirtualLock((LPVOID)addr, len))
        ythrow yexception() << LastSystemErrorText();
#endif
}

void UnlockMemory(const void* addr, size_t len) {
#if defined(_unix_)
    if (munlock(addr, len))
        ythrow yexception() << LastSystemErrorText();
#elif defined(_win_)
    HANDLE hndl = GetCurrentProcess();
    SIZE_T min, max;
    if (!GetProcessWorkingSetSize(hndl, &min, &max))
        ythrow yexception() << LastSystemErrorText();
    if (!SetProcessWorkingSetSize(hndl, min - len, max - len))
        ythrow yexception() << LastSystemErrorText();
    if (!VirtualUnlock((LPVOID)addr, len))
        ythrow yexception() << LastSystemErrorText();
#endif
}

void LockAllMemory(ELockAllMemoryFlags flags) {
    Y_UNUSED(flags);
#if defined(_android_)
// unimplemented
#elif defined(_cygwin_)
// unimplemented
#elif defined(_unix_)
    int sys_flags = 0;
    if (flags & LockCurrentMemory) {
        sys_flags |= MCL_CURRENT;
    }
    if (flags & LockFutureMemory) {
        sys_flags |= MCL_FUTURE;
    }
    if (flags & LockMemoryOnFault) {
        sys_flags |= MCL_ONFAULT;
    }
    if (mlockall(sys_flags)) {
        ythrow yexception() << LastSystemErrorText();
    }
#endif
}

void UnlockAllMemory() {
#if defined(_cygwin_)
// unimplemented
#elif defined(_unix_)
    if (munlockall()) {
        ythrow yexception() << LastSystemErrorText();
    }
#endif
}
