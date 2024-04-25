#include "memory_tag.h"

#include <library/cpp/yt/misc/tls.h>

#include <util/system/compiler.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

YT_DEFINE_THREAD_LOCAL(TMemoryTag, CurrentMemoryTag);

Y_WEAK TMemoryTag GetCurrentMemoryTag()
{
    return CurrentMemoryTag();
}

Y_WEAK void SetCurrentMemoryTag(TMemoryTag tag)
{
    CurrentMemoryTag() = tag;
}

Y_WEAK size_t GetMemoryUsageForTag(TMemoryTag /*tag*/)
{
    return 0;
}

Y_WEAK void GetMemoryUsageForTags(const TMemoryTag* /*tags*/, size_t /*count*/, size_t* /*results*/)
{ }

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

