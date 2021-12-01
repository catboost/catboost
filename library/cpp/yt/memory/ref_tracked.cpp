#include "ref_tracked.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

Y_WEAK TRefCountedTypeCookie TRefCountedTrackerFacade::GetCookie(
    TRefCountedTypeKey /*typeKey*/,
    size_t /*instanceSize*/,
    const TSourceLocation& /*location*/)
{
    return NullRefCountedTypeCookie;
}

Y_WEAK void TRefCountedTrackerFacade::AllocateInstance(TRefCountedTypeCookie /*cookie*/)
{ }

Y_WEAK void TRefCountedTrackerFacade::FreeInstance(TRefCountedTypeCookie /*cookie*/)
{ }

Y_WEAK void TRefCountedTrackerFacade::AllocateTagInstance(TRefCountedTypeCookie /*cookie*/)
{ }

Y_WEAK void TRefCountedTrackerFacade::FreeTagInstance(TRefCountedTypeCookie /*cookie*/)
{ }

Y_WEAK void TRefCountedTrackerFacade::AllocateSpace(TRefCountedTypeCookie /*cookie*/, size_t /*size*/)
{ }

Y_WEAK void TRefCountedTrackerFacade::FreeSpace(TRefCountedTypeCookie /*cookie*/, size_t /*size*/)
{ }

Y_WEAK void TRefCountedTrackerFacade::Dump()
{ }

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
