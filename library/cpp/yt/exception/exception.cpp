#include "exception.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TSimpleException::TSimpleException(TString message)
    : Message_(std::move(message))
{ }

const TString& TSimpleException::GetMessage() const
{
    return Message_;
}

const char* TSimpleException::what() const noexcept
{
    return Message_.c_str();
}

////////////////////////////////////////////////////////////////////////////////

TCompositeException::TCompositeException(TString message)
    : TSimpleException(std::move(message))
    , What_(Message_)
{ }

TCompositeException::TCompositeException(
    const std::exception& exception,
    TString message)
    : TSimpleException(message)
    , InnerException_(std::current_exception())
    , What_(message + "\n" + exception.what())
{ }

const std::exception_ptr& TCompositeException::GetInnerException() const
{
    return InnerException_;
}

const char* TCompositeException::what() const noexcept
{
    return What_.c_str();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
