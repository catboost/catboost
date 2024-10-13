#include "string_builder.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TDelimitedStringBuilderWrapper::TDelimitedStringBuilderWrapper(
    TStringBuilderBase* builder,
    TStringBuf delimiter)
    : Builder_(builder)
    , Delimiter_(delimiter)
{ }

TStringBuilderBase* TDelimitedStringBuilderWrapper::operator->()
{
    return operator&();
}

TStringBuilderBase* TDelimitedStringBuilderWrapper::operator&()
{
    if (!FirstCall_) {
        Builder_->AppendString(Delimiter_);
    }
    FirstCall_ = false;
    return Builder_;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
