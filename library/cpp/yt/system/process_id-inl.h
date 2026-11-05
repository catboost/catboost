#ifndef PROCESS_ID_INL_H_
#error "Direct inclusion of this file is not allowed, include process_id.h"
// For the sake of sane code completion.
#include "process_id.h"
#endif

#include <atomic>

#include <util/system/compiler.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

TProcessId GetProcessIdImpl();

} // namespace NDetail

extern std::atomic<TProcessId> CachedProcessId;

inline TProcessId GetProcessId()
{
    auto cachedProcessId = CachedProcessId.load(std::memory_order::relaxed);
    if (cachedProcessId == InvalidProcessId) [[unlikely]] {
        cachedProcessId = NDetail::GetProcessIdImpl();
        CachedProcessId.store(cachedProcessId, std::memory_order::relaxed);
    }
    return cachedProcessId;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
