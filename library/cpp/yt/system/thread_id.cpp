#include "thread_id.h"

#include <util/system/thread.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

YT_DEFINE_THREAD_LOCAL(TSequentialThreadId, CachedSequentialThreadId, InvalidSequentialThreadId);
std::atomic<TSequentialThreadId> SequentialThreadIdGenerator = InvalidSequentialThreadId;

TSystemThreadId GetSystemThreadId()
{
    static_assert(std::is_same_v<TSystemThreadId, ::TThread::TId>);
    return ::TThread::CurrentThreadNumericId();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
