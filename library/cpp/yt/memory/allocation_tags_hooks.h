#pragma once

#include "allocation_tags.h"

#include <library/cpp/yt/memory/range.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct TAllocationTagsHooks
{
    void* (*CreateAllocationTags)();
    void* (*CopyAllocationTags)(void* opaque);
    void (*DestroyAllocationTags)(void* opaque);
    TRange<TAllocationTag> (*ReadAllocationTags)(void* opaque);
};

const TAllocationTagsHooks& GetAllocationTagsHooks();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
