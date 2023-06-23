#pragma once

#include "defaults.h"

#include <util/generic/flags.h>

enum EProtectMemoryMode {
    PM_NONE = 0x00,  // no access allowed
    PM_READ = 0x01,  // read access allowed
    PM_WRITE = 0x02, // write access allowed
    PM_EXEC = 0x04   // execute access allowed
};

Y_DECLARE_FLAGS(EProtectMemory, EProtectMemoryMode);
Y_DECLARE_OPERATORS_FOR_FLAGS(EProtectMemory);

/**
 * Set protection mode on memory block
 * @param addr Block address to be protected
 * @param length Block size in bytes
 * @param mode A bitwise combination of @c EProtectMemoryMode flags
 * @note On Windows there is no write-only protection mode,
 * so PM_WRITE will be translated to (PM_READ | PM_WRITE) on Windows.
 **/
void ProtectMemory(void* addr, const size_t length, const EProtectMemory mode);
