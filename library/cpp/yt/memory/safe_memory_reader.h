#pragma once

#include "public.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Enables safe read-only access to the address space of the current process.
/*!
 *   Inaccessible memory locations will not cause any traps but rather report
 *   failure via return value.
 */
class TSafeMemoryReader
{
public:
    TSafeMemoryReader();
    TSafeMemoryReader(const TSafeMemoryReader&) = delete;
    ~TSafeMemoryReader();

    //! Attempts to read #value at address #addr.
    //! Returns |true| on success, |false| on failure.
    template <class T>
    bool Read(const void* addr, T* value);

private:
    int FD_ = -1;

    bool ReadRaw(const void* addr, void* ptr, size_t size);
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define SAFE_MEMORY_READER_INL_H_
#include "safe_memory_reader-inl.h"
#undef SAFE_MEMORY_READER_INL_H_
