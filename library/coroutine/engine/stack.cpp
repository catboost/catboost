#include "stack.h"

#include <util/system/defaults.h>
#include <util/system/protect.h>

#include <stdlib.h>
#include <stdio.h>

#ifndef _win_
#   include <unistd.h>
#endif

void TProtectedContStackAllocator::Protect(void* ptr, size_t len) noexcept {
    ProtectMemory(ptr, len, PM_NONE);
}

void TProtectedContStackAllocator::UnProtect(void* ptr, size_t len) noexcept {
    ProtectMemory(ptr, len, PM_READ | PM_WRITE);
}

void TContStackAllocator::TStackType::FailStackOverflow() {
    // Not using FAIL or Cerr, because we should crash
    // as fast as possible, because memory is corrupted
    // and there's high chance that other thread crashes
    // before this message printed.
    static const char message[] = "stack corrupted\n";
#ifdef _win_
    fputs(message, stderr);
#else
    ssize_t res = write(STDERR_FILENO, message, sizeof(message));
    Y_UNUSED(res);
#endif
    abort();
}
