#pragma once

#include "format.h"

#include <util/generic/string.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Appends a certain delimiter starting from the second call.
class TDelimitedStringBuilderWrapper
    : private TNonCopyable
{
public:
    TDelimitedStringBuilderWrapper(
        TStringBuilderBase* builder,
        TStringBuf delimiter = TStringBuf(", "))
        : Builder_(builder)
        , Delimiter_(delimiter)
    { }

    TStringBuilderBase* operator->()
    {
        if (!FirstCall_) {
            Builder_->AppendString(Delimiter_);
        }
        FirstCall_ = false;
        return Builder_;
    }

private:
    TStringBuilderBase* const Builder_;
    const TStringBuf Delimiter_;

    bool FirstCall_ = true;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
