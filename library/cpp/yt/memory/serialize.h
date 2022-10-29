#pragma once

#include "new.h"

////////////////////////////////////////////////////////////////////////////////

template <class T>
class TSerializer<NYT::TIntrusivePtr<T>>
{
public:
    static inline void Save(IOutputStream* out, const NYT::TIntrusivePtr<T>& ptr)
    {
        bool hasValue = ptr.operator bool();
        ::Save(out, hasValue);
        if (hasValue) {
            ::Save(out, *ptr);
        }
    }

    static inline void Load(IInputStream* in, NYT::TIntrusivePtr<T>& ptr)
    {
        bool hasValue;
        ::Load(in, hasValue);
        if (hasValue) {
            auto tmp = NYT::New<T>();
            ::Load(in, *tmp);
            ptr = std::move(tmp);
        } else {
            ptr.Reset();
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
