#pragma once

#include <typeinfo>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////
// Enables accessing type_info for incomplete types.

//! Place this macro in header file after forward-declaring a type.
#define YT_DECLARE_TYPEID(type)

//! Place this macro in header or source file after fully defining a type.
#define YT_DEFINE_TYPEID(type)

//! Equivalent to |typeid(T)| but also works for incomplete types
//! annotated with YT_DECLARE_TYPEID.
template <class T>
const std::type_info& Typeid();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define TYPEID_INL_H_
#include "typeid-inl.h"
#undef TYPEID_INL_H_
