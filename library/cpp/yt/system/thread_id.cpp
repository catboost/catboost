#include "thread_id.h"

#include <util/system/thread.h>

#ifdef _unix_
#include <library/cpp/yt/misc/static_initializer.h>

#include <pthread.h>
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

YT_DEFINE_THREAD_LOCAL(TSystemThreadId, CachedSystemThreadId, InvalidSystemThreadId);

YT_DEFINE_THREAD_LOCAL(TSequentialThreadId, CachedSequentialThreadId, InvalidSequentialThreadId);
std::atomic<TSequentialThreadId> SequentialThreadIdGenerator = InvalidSequentialThreadId;

namespace NDetail {

TSystemThreadId GetSystemThreadIdImpl()
{
    static_assert(std::is_same_v<TSystemThreadId, ::TThread::TId>);
    return ::TThread::CurrentThreadNumericId();
}

} // namespace NDetail

#ifdef _unix_
// The kernel tid is stable for the lifetime of a thread, so we cache it to
// avoid the |gettid| syscall on each call. After a |fork|, however, the
// surviving (calling) thread of the child gets a fresh kernel tid, so the
// cache must be invalidated there.
YT_STATIC_INITIALIZER(
    ::pthread_atfork(
        /*prepare*/ nullptr,
        /*parent*/  nullptr,
        /*child*/   [] { CachedSystemThreadId() = InvalidSystemThreadId; }));
#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
