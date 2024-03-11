#ifndef THREAD_ID_INL_H_
#error "Direct inclusion of this file is not allowed, include thread_id.h"
// For the sake of sane code completion.
#include "thread_id.h"
#endif

#include <library/cpp/yt/misc/tls.h>

#include <atomic>

#include <util/system/compiler.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

extern YT_THREAD_LOCAL(TSequentialThreadId) CachedSequentialThreadId;
extern std::atomic<TSequentialThreadId> SequentialThreadIdGenerator;

inline TSequentialThreadId GetSequentialThreadId()
{
    if (Y_UNLIKELY(CachedSequentialThreadId == InvalidSequentialThreadId)) {
        CachedSequentialThreadId = ++SequentialThreadIdGenerator;
    }
    return CachedSequentialThreadId;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
