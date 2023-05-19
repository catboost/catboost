#include "memory_tag.h"

#include <util/system/compiler.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

Y_WEAK TMemoryTag GetCurrentMemoryTag()
{
    return NullMemoryTag;
}

Y_WEAK void SetCurrentMemoryTag(TMemoryTag /*tag*/)
{ }

Y_WEAK size_t GetMemoryUsageForTag(TMemoryTag /*tag*/)
{
    return 0;
}

Y_WEAK void GetMemoryUsageForTags(const TMemoryTag* /*tags*/, size_t /*count*/, size_t* /*results*/)
{ }

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

