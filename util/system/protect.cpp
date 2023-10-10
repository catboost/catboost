#include "protect.h"

#include <util/generic/yexception.h>
#include <util/generic/string.h>

#include "yassert.h"

#if defined(_unix_) || defined(_darwin_)
    #include <sys/mman.h>
#endif

#ifdef _win_
    #include <Windows.h>
#endif

static TString ModeToString(const EProtectMemory mode) {
    TString strMode;
    if (mode == PM_NONE) {
        return "PM_NONE";
    }

    if (mode & PM_READ) {
        strMode += "PM_READ|";
    }
    if (mode & PM_WRITE) {
        strMode += "PM_WRITE|";
    }
    if (mode & PM_EXEC) {
        strMode += "PM_EXEC|";
    }
    return strMode.substr(0, strMode.size() - 1);
}

void ProtectMemory(void* addr, const size_t length, const EProtectMemory mode) {
    Y_ABORT_UNLESS(!(mode & ~(PM_READ | PM_WRITE | PM_EXEC)), "Invalid memory protection flag combination. ");

#if defined(_unix_) || defined(_darwin_)
    int mpMode = PROT_NONE;
    if (mode & PM_READ) {
        mpMode |= PROT_READ;
    }
    if (mode & PM_WRITE) {
        mpMode |= PROT_WRITE;
    }
    if (mode & PM_EXEC) {
        mpMode |= PROT_EXEC;
    }
    // some old manpages for mprotect say 'const void* addr', but that's wrong
    if (mprotect(addr, length, mpMode) == -1) {
        ythrow TSystemError() << "Memory protection failed for mode " << ModeToString(mode) << ". ";
    }
#endif

#ifdef _win_
    DWORD mpMode = PAGE_NOACCESS;
    // windows developers are not aware of bit flags :(

    /*
     * It's unclear that we should NOT fail on Windows that does not support write-only
     * memory protection. As we don't know, what behavior is more correct, we choose
     * one of them. A discussion was here: REVIEW: 39725
     */
    switch (mode.ToBaseType()) {
        case PM_READ:
            mpMode = PAGE_READONLY;
            break;
        case PM_WRITE:
            mpMode = PAGE_READWRITE;
            break; // BUG: no write-only support
        /*case PM_WRITE:
            ythrow TSystemError() << "Write-only protection mode is not supported under Windows. ";*/
        case PM_READ | PM_WRITE:
            mpMode = PAGE_READWRITE;
            break;
        case PM_EXEC:
            mpMode = PAGE_EXECUTE;
            break;
        case PM_READ | PM_EXEC:
            mpMode = PAGE_EXECUTE_READ;
            break;
        case PM_WRITE | PM_EXEC:
            mpMode = PAGE_EXECUTE_READWRITE;
            break; // BUG: no write-only support
        /*case PM_WRITE | PM_EXEC:
            ythrow TSystemError() << "Write-execute-only protection mode is not supported under Windows. ";*/
        case PM_READ | PM_WRITE | PM_EXEC:
            mpMode = PAGE_EXECUTE_READWRITE;
            break;
    }
    DWORD oldMode = 0;
    if (!VirtualProtect(addr, length, mpMode, &oldMode))
        ythrow TSystemError() << "Memory protection failed for mode " << ModeToString(mode) << ". ";
#endif
}
