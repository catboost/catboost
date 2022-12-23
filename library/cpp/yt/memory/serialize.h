#pragma once

#include "intrusive_ptr.h"

#include <util/ysaveload.h>

////////////////////////////////////////////////////////////////////////////////

template <class T>
class TSerializer<NYT::TIntrusivePtr<T>>
{
public:
    static inline void Save(IOutputStream* output, const NYT::TIntrusivePtr<T>& ptr);
    static inline void Load(IInputStream* input, NYT::TIntrusivePtr<T>& ptr);
};

////////////////////////////////////////////////////////////////////////////////

#define SERIALIZE_PTR_INL_H_
#include "serialize-inl.h"
#undef SERIALIZE_PTR_INL_H_
