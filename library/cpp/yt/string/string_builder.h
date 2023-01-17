#pragma once

#include <util/generic/string.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// Forward declarations.
class TStringBuilderBase;
class TStringBuilder;
class TDelimitedStringBuilderWrapper;

template <size_t Length, class... TArgs>
void Format(TStringBuilderBase* builder, const char (&format)[Length], TArgs&&... args);
template <class... TArgs>
void Format(TStringBuilderBase* builder, TStringBuf format, TArgs&&... args);

////////////////////////////////////////////////////////////////////////////////

//! A simple helper for constructing strings by a sequence of appends.
class TStringBuilderBase
{
public:
    virtual ~TStringBuilderBase() = default;

    char* Preallocate(size_t size);

    void Reserve(size_t size);

    size_t GetLength() const;

    TStringBuf GetBuffer() const;

    void Advance(size_t size);

    void AppendChar(char ch);
    void AppendChar(char ch, int n);

    void AppendString(TStringBuf str);
    void AppendString(const char* str);

    template <size_t Length, class... TArgs>
    void AppendFormat(const char (&format)[Length], TArgs&&... args);
    template <class... TArgs>
    void AppendFormat(TStringBuf format, TArgs&&... args);

    void Reset();

protected:
    char* Begin_ = nullptr;
    char* Current_ = nullptr;
    char* End_ = nullptr;

    virtual void DoReset() = 0;
    virtual void DoReserve(size_t newLength) = 0;

    static constexpr size_t MinBufferLength = 128;
};

////////////////////////////////////////////////////////////////////////////////

class TStringBuilder
    : public TStringBuilderBase
{
public:
    TString Flush();

protected:
    TString Buffer_;

    void DoReset() override;
    void DoReserve(size_t size) override;
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
TString ToStringViaBuilder(const T& value, TStringBuf spec = TStringBuf("v"));

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

#define STRING_BUILDER_INL_H_
#include "string_builder-inl.h"
#undef STRING_BUILDER_INL_H_
