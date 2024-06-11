#ifndef STRING_BUILDER_INL_H_
#error "Direct inclusion of this file is not allowed, include string.h"
// For the sake of sane code completion.
#include "string_builder.h"
#endif

#include <library/cpp/yt/assert/assert.h>

#include <util/stream/str.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

inline char* TStringBuilderBase::Preallocate(size_t size)
{
    Reserve(size + GetLength());
    return Current_;
}

inline void TStringBuilderBase::Reserve(size_t size)
{
    if (Y_UNLIKELY(End_ - Begin_ < static_cast<ssize_t>(size))) {
        size_t length = GetLength();
        auto newLength = std::max(size, MinBufferLength);
        DoReserve(newLength);
        Current_ = Begin_ + length;
    }
}

inline size_t TStringBuilderBase::GetLength() const
{
    return Current_ ? Current_ - Begin_ : 0;
}

inline TStringBuf TStringBuilderBase::GetBuffer() const
{
    return TStringBuf(Begin_, Current_);
}

inline void TStringBuilderBase::Advance(size_t size)
{
    Current_ += size;
    YT_ASSERT(Current_ <= End_);
}

inline void TStringBuilderBase::AppendChar(char ch)
{
    *Preallocate(1) = ch;
    Advance(1);
}

inline void TStringBuilderBase::AppendChar(char ch, int n)
{
    YT_ASSERT(n >= 0);
    if (Y_LIKELY(0 != n)) {
        char* dst = Preallocate(n);
        ::memset(dst, ch, n);
        Advance(n);
    }
}

inline void TStringBuilderBase::AppendString(TStringBuf str)
{
    if (Y_LIKELY(str)) {
        char* dst = Preallocate(str.length());
        ::memcpy(dst, str.begin(), str.length());
        Advance(str.length());
    }
}

inline void TStringBuilderBase::AppendString(const char* str)
{
    AppendString(TStringBuf(str));
}

inline void TStringBuilderBase::Reset()
{
    Begin_ = Current_ = End_ = nullptr;
    DoReset();
}

template <class... TArgs>
void TStringBuilderBase::AppendFormat(TStringBuf format, TArgs&& ... args)
{
    Format(this, TRuntimeFormat{format}, std::forward<TArgs>(args)...);
}

template <size_t Length, class... TArgs>
void TStringBuilderBase::AppendFormat(const char (&format)[Length], TArgs&& ... args)
{
    Format(this, TRuntimeFormat{format}, std::forward<TArgs>(args)...);
}

////////////////////////////////////////////////////////////////////////////////

inline TString TStringBuilder::Flush()
{
    Buffer_.resize(GetLength());
    auto result = std::move(Buffer_);
    Reset();
    return result;
}

inline void TStringBuilder::DoReset()
{
    Buffer_ = {};
}

inline void TStringBuilder::DoReserve(size_t newLength)
{
    Buffer_.ReserveAndResize(newLength);
    auto capacity = Buffer_.capacity();
    Buffer_.ReserveAndResize(capacity);
    Begin_ = &*Buffer_.begin();
    End_ = Begin_ + capacity;
}

////////////////////////////////////////////////////////////////////////////////

inline void FormatValue(TStringBuilderBase* builder, const TStringBuilder& value, TStringBuf /*spec*/)
{
    builder->AppendString(value.GetBuffer());
}

template <class T>
TString ToStringViaBuilder(const T& value, TStringBuf spec)
{
    TStringBuilder builder;
    FormatValue(&builder, value, spec);
    return builder.Flush();
}

////////////////////////////////////////////////////////////////////////////////

// Compatibility for users of NYT::ToString(nyt_type).
template <CFormattable T>
TString ToString(const T& t)
{
    return ToStringViaBuilder(t);
}

// Sometime we want to implement
// FormatValue using util's ToString
// However, if we inside the FormatValue
// we cannot just call the ToString since
// in this scope T is already CFormattable
// and ToString will call the very
// FormatValue we are implementing,
// causing an infinite recursion loop.
// This method is basically a call to
// util's ToString default implementation.
template <class T>
TString ToStringIgnoringFormatValue(const T& t)
{
    TString s;
    ::TStringOutput o(s);
    o << t;
    return s;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#include <util/string/cast.h>

// util/string/cast.h extension for yt and std types only
// TODO(arkady-e1ppa): Abolish ::ToString in
// favour of either NYT::ToString or
// automatic formatting wherever it is needed.
namespace NPrivate {

template <class T>
    requires (
        (NYT::NDetail::IsNYTName<T>() ||
        NYT::NDetail::IsStdName<T>()) &&
        NYT::CFormattable<T>)
struct TToString<T, false>
{
    static TString Cvt(const T& t)
    {
        return NYT::ToStringViaBuilder(t);
    }
};

} // namespace NPrivate
