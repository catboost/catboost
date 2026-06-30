#include "exception.h"

#include <library/cpp/yt/assert/assert.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace {

template <class TRange>
void AddAttributes(TSimpleException::TAttributes& attrs, TRange&& range)
{
    for (auto&& [key, value] : range) {
        YT_VERIFY(attrs.emplace(std::move(key), std::move(value)).second);
    }
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

TSimpleException::TSimpleException(std::string message)
    : Message_(std::move(message))
    , What_(Message_)
{ }

TSimpleException::TSimpleException(
    const std::exception& exception,
    std::string message)
    : InnerException_(std::current_exception())
    , Message_(std::move(message))
    , What_(Message_ + "\n" + exception.what())
{ }

const std::exception_ptr& TSimpleException::GetInnerException() const
{
    return InnerException_;
}

const char* TSimpleException::what() const noexcept
{
    return What_.c_str();
}

const std::string& TSimpleException::GetMessage() const
{
    return Message_;
}

const TSimpleException::TAttributes& TSimpleException::GetAttributes() const &
{
    return Attributes_;
}

TSimpleException::TAttributes&& TSimpleException::GetAttributes() &&
{
    return std::move(Attributes_);
}

TSimpleException& TSimpleException::operator<<= (TExceptionAttribute&& attribute) &
{
    YT_VERIFY(Attributes_.emplace(std::move(attribute.Key), std::move(attribute.Value)).second);
    return *this;
}

TSimpleException& TSimpleException::operator<<= (std::vector<TExceptionAttribute>&& attributes) &
{
    AddAttributes(Attributes_, std::move(attributes));
    return *this;
}

TSimpleException& TSimpleException::operator<<= (TAttributes&& attributes) &
{
    AddAttributes(Attributes_, std::move(attributes));
    return *this;
}

TSimpleException& TSimpleException::operator<<= (const TExceptionAttribute& attribute) &
{
    YT_VERIFY(Attributes_.emplace(attribute.Key, attribute.Value).second);
    return *this;
}

TSimpleException& TSimpleException::operator<<= (const std::vector<TExceptionAttribute>& attributes) &
{
    AddAttributes(Attributes_, attributes);
    return *this;
}

TSimpleException& TSimpleException::operator<<= (const TAttributes& attributes) &
{
    AddAttributes(Attributes_, attributes);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
