#pragma once

#include "defaults.h"

#include <util/generic/flags.h>

// on some systems (not win, freebd, linux, but darwin (Mac OS X)
// multiple mlock calls on the same address range
// require the corresponding number of munlock calls to actually unlock the pages

// on some systems you must have privilege and resource limit

void LockMemory(const void* addr, size_t len);
void UnlockMemory(const void* addr, size_t len);

enum ELockAllMemoryFlag {
    /** Lock all pages which are currently mapped into the address space of the process. */
    LockCurrentMemory = 1,

    /** Lock all pages which will become mapped into the address space of the process in the future. */
    LockFutureMemory = 2,

    /** Since Linux 4.4, with LockCurrentMemory or LockFutureMemory or both, lock only pages that are or once they are present in memory. */
    LockMemoryOnFault = 4,
};
Y_DECLARE_FLAGS(ELockAllMemoryFlags, ELockAllMemoryFlag);
Y_DECLARE_OPERATORS_FOR_FLAGS(ELockAllMemoryFlags);

/**
 * Performs provided locking operation.
 *
 * Does nothing on windows.
 *
 * \param flags                         Locking operation to perform.
 */
void LockAllMemory(ELockAllMemoryFlags flags);

/**
 * Unlocks whatever was locked with a previous call to `LockAllMemory`.
 *
 * Does nothing on windows.
 */
void UnlockAllMemory();
