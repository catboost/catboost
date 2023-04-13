#ifndef SAFE_MEMORY_READER_INL_H_
#error "Direct inclusion of this file is not allowed, include safe_memory_reader.h"
// For the sake of sane code completion.
#include "safe_memory_reader.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
bool TSafeMemoryReader::Read(const void* addr, T* value)
{
    return ReadRaw(addr, value, sizeof(*value));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
