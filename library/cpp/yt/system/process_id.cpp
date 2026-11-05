#include "process_id.h"

#ifdef _unix_
#include <library/cpp/yt/misc/static_initializer.h>

#include <pthread.h>
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

std::atomic<TProcessId> CachedProcessId = InvalidProcessId;

namespace NDetail {

TProcessId GetProcessIdImpl()
{
    return ::GetPID();
}

} // namespace NDetail

#ifdef _unix_
// The pid is stable for the lifetime of a process, so we cache it to avoid the
// |getpid| syscall on each call. After a |fork|, however, the child runs with a
// fresh pid, so the cache must be invalidated there.
YT_STATIC_INITIALIZER(
    ::pthread_atfork(
        /*prepare*/ nullptr,
        /*parent*/  nullptr,
        /*child*/   [] { CachedProcessId.store(InvalidProcessId, std::memory_order::relaxed); }));
#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
