#pragma once
#ifndef ACCESS_SPREADER_INL_H_
#error "Direct inclusion of this file is not allowed, include access_spreader.h"
#endif

#include <library/cpp/yt/assert/assert.h>

#include <util/system/compiler.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

ui32 GetAccessStripeSeed();

} // namespace NDetail

Y_FORCE_INLINE int GetCurrentAccessStripe(int stripeCount)
{
    YT_ASSERT(stripeCount > 0);
    return static_cast<int>(NDetail::GetAccessStripeSeed() % static_cast<ui32>(stripeCount));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
