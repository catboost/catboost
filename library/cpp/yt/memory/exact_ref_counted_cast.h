#pragma once

#include "ref_counted.h"

#include <type_traits>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! A downcast that succeeds iff the object was allocated as *exactly* |New<T>()|
//! (not a subclass, not some other allocation path).
/*!
 *  |New<T>()| does not construct a |T| but a final |TRefCountedWrapper<T>|
 *  deriving from |T|, so |T| is never a leaf and |dynamic_cast<T*>| takes the
 *  slow is-a path. This casts to the wrapper instead and upcasts back to |T*|;
 *  since the wrapper is final, dynamic_cast lowers to one type_info compare --
 *  O(1), as cheap on a miss as on a hit.
 *
 *  Pointer form returns |nullptr| on a miss (or null operand); reference form
 *  throws |std::bad_cast|. Constness is preserved. |T| must derive from
 *  |TRefCountedBase| and |U| must be polymorphic. Requires RTTI.
 */
template <class T, class U>
[[nodiscard]] std::conditional_t<std::is_const_v<U>, const T, T>* ExactRefCountedCast(U* ptr) noexcept;

template <class T, class U>
[[nodiscard]] std::conditional_t<std::is_const_v<U>, const T, T>& ExactRefCountedCast(U& ref);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define EXACT_REF_COUNTED_CAST_INL_H_
#include "exact_ref_counted_cast-inl.h"
#undef EXACT_REF_COUNTED_CAST_INL_H_
