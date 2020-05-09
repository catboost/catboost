#pragma once

#include <util/generic/string.h>

namespace NJson {
    namespace NImpl {
        enum EType {
            ARRAY,
            MAP,
            MAP_KEY
        };
    }
    template <class TStringType>
    struct TPathElemImpl {
        NImpl::EType Type;
        TStringType Key;
        int ArrayCounter;

        TPathElemImpl(NImpl::EType type)
            : Type(type)
            , ArrayCounter()
        {
        }

        TPathElemImpl(const TStringType& key)
            : Type(NImpl::MAP_KEY)
            , Key(key)
            , ArrayCounter()
        {
        }

        TPathElemImpl(int arrayCounter)
            : Type(NImpl::ARRAY)
            , ArrayCounter(arrayCounter)
        {
        }
    };

    typedef TPathElemImpl<TString> TPathElem;
}
