#pragma once

#include "attributes.h"

#include <util/generic/hash.h>

#include <exception>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////
// This is poor man's version of NYT::TErrorException to be used in
// a limited subset of core libraries that are needed to implement NYT::TError.

class TSimpleException
    : public std::exception
{
public:
    using TAttributes = THashMap<
        TExceptionAttribute::TKey,
        TExceptionAttribute::TValue>;

    template <class TValue>
    static constexpr bool CNestable = requires (TSimpleException& ex, TValue&& operand) {
        { ex <<= std::forward<TValue>(operand) } -> std::same_as<TSimpleException&>;
    };

    explicit TSimpleException(TString message);
    TSimpleException(
        const std::exception& exception,
        TString message);

    const std::exception_ptr& GetInnerException() const;
    const char* what() const noexcept override;

    const TString& GetMessage() const;

    const TAttributes& GetAttributes() const &;
    TAttributes&& GetAttributes() &&;

    TSimpleException& operator<<= (TExceptionAttribute&& attribute) &;
    TSimpleException& operator<<= (std::vector<TExceptionAttribute>&& attributes) &;
    TSimpleException& operator<<= (TAttributes&& attributes) &;

    TSimpleException& operator<<= (const TExceptionAttribute& attribute) &;
    TSimpleException& operator<<= (const std::vector<TExceptionAttribute>& attributes) &;
    TSimpleException& operator<<= (const TAttributes& attributes) &;

    // NB: clang is incapable of parsing such requirements (which refer back to the class) out-of-line.
    // To keep this overload from winning in resolution
    // when constraint actually fails, we define method right here.
    template <class TValue>
        requires CNestable<TValue>
    TSimpleException&& operator<< (TValue&& value) &&
    {
        return std::move(*this <<= std::forward<TValue>(value));
    }

    template <class TValue>
        requires CNestable<TValue>
    TSimpleException operator<< (TValue&& value) const &
    {
        return TSimpleException(*this) << std::forward<TValue>(value);
    }

private:
    const std::exception_ptr InnerException_;
    const TString Message_;
    const TString What_;

    TAttributes Attributes_;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
