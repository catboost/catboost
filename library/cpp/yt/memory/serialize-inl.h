#ifndef SERIALIZE_PTR_INL_H_
#error "Direct inclusion of this file is not allowed, include serialize.h"
// For the sake of sane code completion.
#include "serialize.h"
#endif

#include "new.h"

////////////////////////////////////////////////////////////////////////////////

template <class T>
void TSerializer<NYT::TIntrusivePtr<T>>::Save(IOutputStream* output, const NYT::TIntrusivePtr<T>& ptr)
{
    bool hasValue = ptr.operator bool();
    ::Save(output, hasValue);
    if (hasValue) {
        ::Save(output, *ptr);
    }
}

template <class T>
void TSerializer<NYT::TIntrusivePtr<T>>::Load(IInputStream* input, NYT::TIntrusivePtr<T>& ptr)
{
    bool hasValue;
    ::Load(input, hasValue);
    if (hasValue) {
        auto tmp = NYT::New<T>();
        ::Load(input, *tmp);
        ptr = std::move(tmp);
    } else {
        ptr.Reset();
    }
}

////////////////////////////////////////////////////////////////////////////////
