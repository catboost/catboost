#pragma once

#include <util/system/compiler.h>
#include <util/system/types.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Returns an access stripe index in [0, |stripeCount|).
//!
//! Values may collide and change between calls; stripes must remain safe for
//! concurrent access. |stripeCount| must be positive.
Y_FORCE_INLINE int GetCurrentAccessStripe(int stripeCount);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ACCESS_SPREADER_INL_H_
#include "access_spreader-inl.h"
#undef ACCESS_SPREADER_INL_H_
