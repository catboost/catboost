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

#include <limits>

void LockMemory(const void* addr, size_t len) {
#if defined(_unix_)
    if (0 == len) {
        return;
    }
    Y_ASSERT(static_cast<ssize_t>(len) > 0);
    const size_t pageSize = NSystemInfo::GetPageSize();
    const char* begin = AlignDown(static_cast<const char*>(addr), pageSize);
    const char* end = AlignUp(static_cast<const char*>(addr) + len, pageSize);
    if (mlock(begin, end - begin)) {
        ythrow yexception() << LastSystemErrorText();
    }
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
    if (0 == len) {
        return;
    }
    Y_ASSERT(static_cast<ssize_t>(len) > 0);
    const size_t pageSize = NSystemInfo::GetPageSize();
    const char* begin = AlignDown(static_cast<const char*>(addr), pageSize);
    const char* end = AlignUp(static_cast<const char*>(addr) + len, pageSize);
    if (munlock(begin, end - begin)) {
        ythrow yexception() << LastSystemErrorText();
    }
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
