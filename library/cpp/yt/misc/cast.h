#pragma once
#include <util/generic/yexception.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TIntegralCastException
    : public yexception
{ };

template <class T, class S>
bool TryIntegralCast(S value, T* result);

template <class T, class S>
T CheckedIntegralCast(S value);

////////////////////////////////////////////////////////////////////////////////

class TEnumCastException
    : public yexception
{ };

template <class T, class S>
bool TryEnumCast(S value, T* result);

template <class T, class S>
T CheckedEnumCast(S value);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define CAST_INL_H_
#include "cast-inl.h"
#undef CAST_INL_H_
