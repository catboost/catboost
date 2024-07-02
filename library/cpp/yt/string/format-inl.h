#ifndef FORMAT_INL_H_
#error "Direct inclusion of this file is not allowed, include format.h"
// For the sake of sane code completion.
#include "format.h"
#endif

#include "guid.h"
#include "enum.h"
#include "string.h"

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/small_containers/compact_vector.h>

#include <library/cpp/yt/containers/enum_indexed_array.h>

#include <library/cpp/yt/misc/concepts.h>
#include <library/cpp/yt/misc/enum.h>
#include <library/cpp/yt/misc/wrapper_traits.h>

#include <util/generic/maybe.h>

#include <util/system/platform.h>

#include <cctype>
#include <optional>
#include <span>

#if __cplusplus >= 202302L
    #include <filesystem>
#endif

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

inline void FormatValue(TStringBuilderBase* builder, const TStringBuilder& value, TStringBuf /*spec*/)
{
    builder->AppendString(value.GetBuffer());
}

////////////////////////////////////////////////////////////////////////////////

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

// Helper functions for formatting.
namespace NDetail {

constexpr inline char IntroductorySymbol = '%';
constexpr inline char GenericSpecSymbol = 'v';

inline bool IsQuotationSpecSymbol(char symbol)
{
    return symbol == 'Q' || symbol == 'q';
}

////////////////////////////////////////////////////////////////////////////////

template <class TValue>
void FormatValueViaSprintf(
    TStringBuilderBase* builder,
    TValue value,
    TStringBuf spec,
    TStringBuf genericSpec);

template <class TValue>
void FormatIntValue(
    TStringBuilderBase* builder,
    TValue value,
    TStringBuf spec,
    TStringBuf genericSpec);

void FormatPointerValue(
    TStringBuilderBase* builder,
    const void* value,
    TStringBuf spec);

////////////////////////////////////////////////////////////////////////////////

// Helper concepts for matching the correct overload.
// NB(arkady-e1ppa): We prefer to hardcode the known types
// so that someone doesn't accidentally implement the
// "SimpleRange" concept and have a non-trivial
// formatting procedure at the same time.

template <class R>
concept CKnownRange =
    requires (R r) { [] <class... Ts> (std::vector<Ts...>) { } (r); } ||
    requires (R r) { [] <class T, size_t E> (std::span<T, E>) { } (r); } ||
    requires (R r) { [] <class T, size_t N> (TCompactVector<T, N>) { } (r); } ||
    requires (R r) { [] <class... Ts> (std::set<Ts...>) { } (r); } ||
    requires (R r) { [] <class... Ts> (THashSet<Ts...>) { } (r); } ||
    requires (R r) { [] <class... Ts> (THashMultiSet<Ts...>) { } (r); };

////////////////////////////////////////////////////////////////////////////////

template <class R>
concept CKnownKVRange =
    requires (R r) { [] <class... Ts> (std::map<Ts...>) { } (r); } ||
    requires (R r) { [] <class... Ts> (std::multimap<Ts...>) { } (r); } ||
    requires (R r) { [] <class... Ts> (THashMap<Ts...>) { } (r); } ||
    requires (R r) { [] <class... Ts> (THashMultiMap<Ts...>) { } (r); };

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <class TRange, class TFormatter>
void FormatRange(TStringBuilderBase* builder, const TRange& range, const TFormatter& formatter, size_t limit = std::numeric_limits<size_t>::max())
{
    builder->AppendChar('[');
    size_t index = 0;
    for (const auto& item : range) {
        if (index > 0) {
            builder->AppendString(DefaultJoinToStringDelimiter);
        }
        if (index == limit) {
            builder->AppendString(DefaultRangeEllipsisFormat);
            break;
        }
        formatter(builder, item);
        ++index;
    }
    builder->AppendChar(']');
}

////////////////////////////////////////////////////////////////////////////////

template <class TRange, class TFormatter>
void FormatKeyValueRange(TStringBuilderBase* builder, const TRange& range, const TFormatter& formatter, size_t limit = std::numeric_limits<size_t>::max())
{
    builder->AppendChar('{');
    size_t index = 0;
    for (const auto& item : range) {
        if (index > 0) {
            builder->AppendString(DefaultJoinToStringDelimiter);
        }
        if (index == limit) {
            builder->AppendString(DefaultRangeEllipsisFormat);
            break;
        }
        formatter(builder, item.first);
        builder->AppendString(DefaultKeyValueDelimiter);
        formatter(builder, item.second);
        ++index;
    }
    builder->AppendChar('}');
}

////////////////////////////////////////////////////////////////////////////////

template <class R>
concept CFormattableRange =
    NDetail::CKnownRange<R> &&
    CFormattable<typename R::value_type>;

template <class R>
concept CFormattableKVRange =
    NDetail::CKnownKVRange<R> &&
    CFormattable<typename R::key_type> &&
    CFormattable<typename R::value_type>;

////////////////////////////////////////////////////////////////////////////////

template <class TRange, class TFormatter>
typename TFormattableView<TRange, TFormatter>::TBegin TFormattableView<TRange, TFormatter>::begin() const
{
    return RangeBegin;
}

template <class TRange, class TFormatter>
typename TFormattableView<TRange, TFormatter>::TEnd TFormattableView<TRange, TFormatter>::end() const
{
    return RangeEnd;
}

template <class TRange, class TFormatter>
TFormattableView<TRange, TFormatter> MakeFormattableView(
    const TRange& range,
    TFormatter&& formatter)
{
    return TFormattableView<TRange, std::decay_t<TFormatter>>{range.begin(), range.end(), std::forward<TFormatter>(formatter)};
}

template <class TRange, class TFormatter>
TFormattableView<TRange, TFormatter> MakeShrunkFormattableView(
    const TRange& range,
    TFormatter&& formatter,
    size_t limit)
{
    return TFormattableView<TRange, std::decay_t<TFormatter>>{
        range.begin(),
        range.end(),
        std::forward<TFormatter>(formatter),
        limit};
}

template <class TFormatter>
TFormatterWrapper<TFormatter> MakeFormatterWrapper(
    TFormatter&& formatter)
{
    return TFormatterWrapper<TFormatter>{
        .Formatter = std::move(formatter)
    };
}

template <class... TArgs>
TLazyMultiValueFormatter<TArgs...>::TLazyMultiValueFormatter(
    TStringBuf format,
    TArgs&&... args)
    : Format_(format)
    , Args_(std::forward<TArgs>(args)...)
{ }

template <class... TArgs>
auto MakeLazyMultiValueFormatter(TStringBuf format, TArgs&&... args)
{
    return TLazyMultiValueFormatter<TArgs...>(format, std::forward<TArgs>(args)...);
}

////////////////////////////////////////////////////////////////////////////////

// Non-container objects.

#define XX(valueType, castType, genericSpec) \
    inline void FormatValue(TStringBuilderBase* builder, valueType value, TStringBuf spec) \
    { \
        NYT::NDetail::FormatIntValue(builder, static_cast<castType>(value), spec, genericSpec); \
    }

XX(i8,                  i32,      TStringBuf("d"))
XX(ui8,                 ui32,     TStringBuf("u"))
XX(i16,                 i32,      TStringBuf("d"))
XX(ui16,                ui32,     TStringBuf("u"))
XX(i32,                 i32,      TStringBuf("d"))
XX(ui32,                ui32,     TStringBuf("u"))
XX(long,                i64,      TStringBuf(PRIdLEAST64))
XX(long long,           i64,      TStringBuf(PRIdLEAST64))
XX(unsigned long,       ui64,     TStringBuf(PRIuLEAST64))
XX(unsigned long long,  ui64,     TStringBuf(PRIuLEAST64))

#undef XX

#define XX(valueType, castType, genericSpec) \
    inline void FormatValue(TStringBuilderBase* builder, valueType value, TStringBuf spec) \
    { \
        NYT::NDetail::FormatValueViaSprintf(builder, static_cast<castType>(value), spec, genericSpec); \
    }

XX(double,              double,   TStringBuf("lf"))
XX(float,               float,    TStringBuf("f"))

#undef XX

// Pointer
template <class T>
void FormatValue(TStringBuilderBase* builder, T* value, TStringBuf spec)
{
    NYT::NDetail::FormatPointerValue(builder, static_cast<const void*>(value), spec);
}

// TStringBuf
inline void FormatValue(TStringBuilderBase* builder, TStringBuf value, TStringBuf spec)
{
    if (!spec) {
        builder->AppendString(value);
        return;
    }

    // Parse alignment.
    bool alignLeft = false;
    const char* current = spec.begin();
    if (*current == '-') {
        alignLeft = true;
        ++current;
    }

    bool hasAlign = false;
    int alignSize = 0;
    while (*current >= '0' && *current <= '9') {
        hasAlign = true;
        alignSize = 10 * alignSize + (*current - '0');
        if (alignSize > 1000000) {
            builder->AppendString(TStringBuf("<alignment overflow>"));
            return;
        }
        ++current;
    }

    int padding = 0;
    bool padLeft = false;
    bool padRight = false;
    if (hasAlign) {
        padding = alignSize - value.size();
        if (padding < 0) {
            padding = 0;
        }
        padLeft = !alignLeft;
        padRight = alignLeft;
    }

    bool singleQuotes = false;
    bool doubleQuotes = false;
    bool escape = false;
    while (current < spec.end()) {
        switch (*current++) {
            case 'q':
                singleQuotes = true;
                break;
            case 'Q':
                doubleQuotes = true;
                break;
            case 'h':
                escape =  true;
                break;
        }
    }

    if (padLeft) {
        builder->AppendChar(' ', padding);
    }

    if (singleQuotes || doubleQuotes || escape) {
        for (const char* valueCurrent = value.begin(); valueCurrent < value.end(); ++valueCurrent) {
            char ch = *valueCurrent;
            if (ch == '\n') {
                builder->AppendString("\\n");
            } else if (ch == '\t') {
                builder->AppendString("\\t");
            } else if (ch == '\\') {
                builder->AppendString("\\\\");
            } else if (ch < PrintableASCIILow || ch > PrintableASCIIHigh) {
                builder->AppendString("\\x");
                builder->AppendChar(IntToHexLowercase[static_cast<ui8>(ch) >> 4]);
                builder->AppendChar(IntToHexLowercase[static_cast<ui8>(ch) & 0xf]);
            } else if ((singleQuotes && ch == '\'') || (doubleQuotes && ch == '\"')) {
                builder->AppendChar('\\');
                builder->AppendChar(ch);
            } else {
                builder->AppendChar(ch);
            }
        }
    } else {
        builder->AppendString(value);
    }

    if (padRight) {
        builder->AppendChar(' ', padding);
    }
}

// TString
inline void FormatValue(TStringBuilderBase* builder, const TString& value, TStringBuf spec)
{
    FormatValue(builder, TStringBuf(value), spec);
}

// const char*
inline void FormatValue(TStringBuilderBase* builder, const char* value, TStringBuf spec)
{
    FormatValue(builder, TStringBuf(value), spec);
}

template <size_t N>
inline void FormatValue(TStringBuilderBase* builder, const char (&value)[N], TStringBuf spec)
{
    FormatValue(builder, TStringBuf(value), spec);
}

// char*
inline void FormatValue(TStringBuilderBase* builder, char* value, TStringBuf spec)
{
    FormatValue(builder, TStringBuf(value), spec);
}

// std::string
inline void FormatValue(TStringBuilderBase* builder, const std::string& value, TStringBuf spec)
{
    FormatValue(builder, TStringBuf(value), spec);
}

// std::string_view
inline void FormatValue(TStringBuilderBase* builder, const std::string_view& value, TStringBuf spec)
{
    FormatValue(builder, TStringBuf(value), spec);
}

#if __cplusplus >= 202302L
// std::filesystem::path
inline void FormatValue(TStringBuilderBase* builder, const std::filesystem::path& value, TStringBuf spec)
{
    FormatValue(builder, std::string(value), spec);
}
#endif

// char
inline void FormatValue(TStringBuilderBase* builder, char value, TStringBuf spec)
{
    FormatValue(builder, TStringBuf(&value, 1), spec);
}

// bool
inline void FormatValue(TStringBuilderBase* builder, bool value, TStringBuf spec)
{
    // Parse custom flags.
    bool lowercase = false;
    const char* current = spec.begin();
    while (current != spec.end()) {
        if (*current == 'l') {
            ++current;
            lowercase = true;
        } else if (NYT::NDetail::IsQuotationSpecSymbol(*current)) {
            ++current;
        } else
            break;
    }

    auto str = lowercase
        ? (value ? TStringBuf("true") : TStringBuf("false"))
        : (value ? TStringBuf("True") : TStringBuf("False"));

    builder->AppendString(str);
}

// TDuration
inline void FormatValue(TStringBuilderBase* builder, TDuration value, TStringBuf /*spec*/)
{
    builder->AppendFormat("%vus", value.MicroSeconds());
}

// TInstant
inline void FormatValue(TStringBuilderBase* builder, TInstant value, TStringBuf spec)
{
    // TODO(babenko): Optimize.
    FormatValue(builder, NYT::ToStringIgnoringFormatValue(value), spec);
}

// Enum
template <class TEnum>
    requires (TEnumTraits<TEnum>::IsEnum)
void FormatValue(TStringBuilderBase* builder, TEnum value, TStringBuf spec)
{
    // Parse custom flags.
    bool lowercase = false;
    const char* current = spec.begin();
    while (current != spec.end()) {
        if (*current == 'l') {
            ++current;
            lowercase = true;
        } else if (NYT::NDetail::IsQuotationSpecSymbol(*current)) {
            ++current;
        } else {
            break;
        }
    }

    FormatEnum(builder, value, lowercase);
}

template <class TArcadiaEnum>
    requires (std::is_enum_v<TArcadiaEnum> && !TEnumTraits<TArcadiaEnum>::IsEnum)
void FormatValue(TStringBuilderBase* builder, TArcadiaEnum value, TStringBuf /*spec*/)
{
    // NB(arkady-e1ppa): This can catch normal enums which
    // just want to be serialized as numbers.
    // Unfortunately, we have no way of determining that other than
    // marking every relevant arcadia enum in the code by trait
    // or writing their complete trait and placing such trait in
    // every single file where it is formatted.
    // We gotta figure something out but until that
    // we will just have to make a string for such enums.
    // If only arcadia enums provided compile-time check
    // if enum is serializable :(((((.
    builder->AppendString(NYT::ToStringIgnoringFormatValue(value));
}

// Container objects.
// NB(arkady-e1ppa): In order to support container combinations
// we forward-declare them before defining.

// TMaybe
template <class T, class TPolicy>
void FormatValue(TStringBuilderBase* builder, const TMaybe<T, TPolicy>& value, TStringBuf spec);

// std::optional
template <CFormattable T>
void FormatValue(TStringBuilderBase* builder, const std::optional<T>& value, TStringBuf spec);

// std::pair
template <CFormattable A, CFormattable B>
void FormatValue(TStringBuilderBase* builder, const std::pair<A, B>& value, TStringBuf spec);

// std::tuple
template <CFormattable... Ts>
void FormatValue(TStringBuilderBase* builder, const std::tuple<Ts...>& value, TStringBuf spec);

// TEnumIndexedArray
template <class E, CFormattable T>
void FormatValue(TStringBuilderBase* builder, const TEnumIndexedArray<E, T>& collection, TStringBuf spec);

// One-valued ranges
template <CFormattableRange TRange>
void FormatValue(TStringBuilderBase* builder, const TRange& collection, TStringBuf spec);

// Two-valued ranges
template <CFormattableKVRange TRange>
void FormatValue(TStringBuilderBase* builder, const TRange& collection, TStringBuf spec);

// FormattableView
template <class TRange, class TFormatter>
void FormatValue(
    TStringBuilderBase* builder,
    const TFormattableView<TRange, TFormatter>& formattableView,
    TStringBuf spec);

// TFormatterWrapper
template <class TFormatter>
void FormatValue(
    TStringBuilderBase* builder,
    const TFormatterWrapper<TFormatter>& wrapper,
    TStringBuf spec);

// TLazyMultiValueFormatter
template <class... TArgs>
void FormatValue(
    TStringBuilderBase* builder,
    const TLazyMultiValueFormatter<TArgs...>& value,
    TStringBuf /*spec*/);

// TMaybe
template <class T, class TPolicy>
void FormatValue(TStringBuilderBase* builder, const TMaybe<T, TPolicy>& value, TStringBuf spec)
{
    FormatValue(builder, NYT::ToStringIgnoringFormatValue(value), spec);
}

// std::optional: nullopt
inline void FormatValue(TStringBuilderBase* builder, std::nullopt_t, TStringBuf /*spec*/)
{
    builder->AppendString(TStringBuf("<null>"));
}

// std::optional: generic T
template <CFormattable T>
void FormatValue(TStringBuilderBase* builder, const std::optional<T>& value, TStringBuf spec)
{
    if (value.has_value()) {
        FormatValue(builder, *value, spec);
    } else {
        FormatValue(builder, std::nullopt, spec);
    }
}

// std::pair
template <CFormattable A, CFormattable B>
void FormatValue(TStringBuilderBase* builder, const std::pair<A, B>& value, TStringBuf spec)
{
    builder->AppendChar('{');
    FormatValue(builder, value.first, spec);
    builder->AppendString(TStringBuf(", "));
    FormatValue(builder, value.second, spec);
    builder->AppendChar('}');
}

// std::tuple
template <CFormattable... Ts>
void FormatValue(TStringBuilderBase* builder, const std::tuple<Ts...>& value, TStringBuf spec)
{
    builder->AppendChar('{');

    [&] <size_t... Idx> (std::index_sequence<Idx...>) {
        ([&] {
            FormatValue(builder, std::get<Idx>(value), spec);
            if constexpr (Idx != sizeof...(Ts)) {
                builder->AppendString(TStringBuf(", "));
            }
        } (), ...);
    } (std::index_sequence_for<Ts...>());

    builder->AppendChar('}');
}

// TEnumIndexedArray
template <class E, CFormattable T>
void FormatValue(TStringBuilderBase* builder, const TEnumIndexedArray<E, T>& collection, TStringBuf spec)
{
    builder->AppendChar('{');
    bool firstItem = true;
    for (const auto& index : TEnumTraits<E>::GetDomainValues()) {
        if (!firstItem) {
            builder->AppendString(DefaultJoinToStringDelimiter);
        }
        FormatValue(builder, index, spec);
        builder->AppendString(": ");
        FormatValue(builder, collection[index], spec);
        firstItem = false;
    }
    builder->AppendChar('}');
}

// One-valued ranges
template <CFormattableRange TRange>
void FormatValue(TStringBuilderBase* builder, const TRange& collection, TStringBuf /*spec*/)
{
    NYT::FormatRange(builder, collection, TDefaultFormatter());
}

// Two-valued ranges
template <CFormattableKVRange TRange>
void FormatValue(TStringBuilderBase* builder, const TRange& collection, TStringBuf /*spec*/)
{
    NYT::FormatKeyValueRange(builder, collection, TDefaultFormatter());
}

// FormattableView
template <class TRange, class TFormatter>
void FormatValue(
    TStringBuilderBase* builder,
    const TFormattableView<TRange, TFormatter>& formattableView,
    TStringBuf /*spec*/)
{
    NYT::FormatRange(builder, formattableView, formattableView.Formatter, formattableView.Limit);
}

// TFormatterWrapper
template <class TFormatter>
void FormatValue(
    TStringBuilderBase* builder,
    const TFormatterWrapper<TFormatter>& wrapper,
    TStringBuf /*spec*/)
{
    wrapper.Formatter(builder);
}

// TLazyMultiValueFormatter
template <class... TArgs>
void FormatValue(
    TStringBuilderBase* builder,
    const TLazyMultiValueFormatter<TArgs...>& value,
    TStringBuf /*spec*/)
{
    std::apply(
        [&] <class... TInnerArgs> (TInnerArgs&&... args) {
            builder->AppendFormat(value.Format_, std::forward<TInnerArgs>(args)...);
        },
        value.Args_);
}

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <size_t HeadPos, class... TArgs>
class TValueFormatter;

template <size_t HeadPos>
class TValueFormatter<HeadPos>
{
public:
    void operator() (size_t /*index*/, TStringBuilderBase* builder, TStringBuf /*spec*/) const
    {
        builder->AppendString(TStringBuf("<missing argument>"));
    }
};

template <size_t HeadPos, class THead, class... TTail>
class TValueFormatter<HeadPos, THead, TTail...>
{
public:
    explicit TValueFormatter(const THead& head, const TTail&... tail) noexcept
        : Head_(head)
        , TailFormatter_(tail...)
    { }

    void operator() (size_t index, TStringBuilderBase* builder, TStringBuf spec) const
    {
        YT_ASSERT(index >= HeadPos);
        if (index == HeadPos) {
            FormatValue(builder, Head_, spec);
        } else {
            TailFormatter_(index, builder, spec);
        }
    }

private:
    const THead& Head_;
    TValueFormatter<HeadPos + 1, TTail...> TailFormatter_;
};

////////////////////////////////////////////////////////////////////////////////

template <class TRangeValue>
class TRangeFormatter
{
public:
    template <class... TArgs>
        requires std::constructible_from<std::span<const TRangeValue>, TArgs...>
    explicit TRangeFormatter(TArgs&&... args) noexcept
        : Span_(std::forward<TArgs>(args)...)
    { }

    void operator() (size_t index, TStringBuilderBase* builder, TStringBuf spec) const
    {
        if (index >= Span_.size()) {
            builder->AppendString(TStringBuf("<missing argument>"));
        } else {
            FormatValue(builder, *(Span_.begin() + index), spec);
        }
    }

private:
    std::span<const TRangeValue> Span_;
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CFormatter = CInvocable<T, void(size_t, TStringBuilderBase*, TStringBuf)>;

////////////////////////////////////////////////////////////////////////////////

template <CFormatter TFormatter>
void RunFormatter(
    TStringBuilderBase* builder,
    TStringBuf format,
    const TFormatter& formatter)
{
    size_t argIndex = 0;
    auto current = std::begin(format);
    auto end = std::end(format);
    while (true) {
        // Scan verbatim part until stop symbol.
        auto verbatimBegin = current;
        auto verbatimEnd = std::find(current, end, IntroductorySymbol);

        // Copy verbatim part, if any.
        size_t verbatimSize = verbatimEnd - verbatimBegin;
        if (verbatimSize > 0) {
            builder->AppendString(TStringBuf(verbatimBegin, verbatimSize));
        }

        // Handle stop symbol.
        current = verbatimEnd;
        if (current == end) {
            break;
        }

        YT_ASSERT(*current == IntroductorySymbol);
        ++current;

        if (*current == IntroductorySymbol) {
            // Verbatim %.
            builder->AppendChar(IntroductorySymbol);
            ++current;
            continue;
        }

        // Scan format part until stop symbol.
        auto argFormatBegin = current;
        auto argFormatEnd = argFormatBegin;
        bool singleQuotes = false;
        bool doubleQuotes = false;

        while (
            argFormatEnd != end &&
            *argFormatEnd != GenericSpecSymbol &&     // value in generic format
            *argFormatEnd != 'd' &&                   // others are standard specifiers supported by printf
            *argFormatEnd != 'i' &&
            *argFormatEnd != 'u' &&
            *argFormatEnd != 'o' &&
            *argFormatEnd != 'x' &&
            *argFormatEnd != 'X' &&
            *argFormatEnd != 'f' &&
            *argFormatEnd != 'F' &&
            *argFormatEnd != 'e' &&
            *argFormatEnd != 'E' &&
            *argFormatEnd != 'g' &&
            *argFormatEnd != 'G' &&
            *argFormatEnd != 'a' &&
            *argFormatEnd != 'A' &&
            *argFormatEnd != 'c' &&
            *argFormatEnd != 's' &&
            *argFormatEnd != 'p' &&
            *argFormatEnd != 'n')
        {
            switch (*argFormatEnd) {
                case 'q':
                    singleQuotes = true;
                    break;
                case 'Q':
                    doubleQuotes = true;
                    break;
                case 'h':
                    break;
            }
            ++argFormatEnd;
        }

        // Handle end of format string.
        if (argFormatEnd != end) {
            ++argFormatEnd;
        }

        // 'n' means 'nothing'; skip the argument.
        if (*argFormatBegin != 'n') {
            // Format argument.
            TStringBuf argFormat(argFormatBegin, argFormatEnd);
            if (singleQuotes) {
                builder->AppendChar('\'');
            }
            if (doubleQuotes) {
                builder->AppendChar('"');
            }
            formatter(argIndex++, builder, argFormat);
            if (singleQuotes) {
                builder->AppendChar('\'');
            }
            if (doubleQuotes) {
                builder->AppendChar('"');
            }
        }

        current = argFormatEnd;
    }
}

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <class... TArgs>
void Format(TStringBuilderBase* builder, TFormatString<TArgs...> format, TArgs&&... args)
{
    // NB(arkady-e1ppa): "if constexpr" is done in order to prevent
    // compiler from emitting "No matching function to call"
    // when arguments are not formattable.
    // Compiler would crash in TFormatString ctor
    // anyway (e.g. program would not compile) but
    // for some reason it does look ahead and emits
    // a second error.
    if constexpr ((CFormattable<TArgs> && ...)) {
        NYT::NDetail::TValueFormatter<0, TArgs...> formatter(args...);
        NYT::NDetail::RunFormatter(builder, format.Get(), formatter);
    }
}

template <class... TArgs>
TString Format(TFormatString<TArgs...> format, TArgs&&... args)
{
    TStringBuilder builder;
    Format(&builder, format, std::forward<TArgs>(args)...);
    return builder.Flush();
}

////////////////////////////////////////////////////////////////////////////////

template <size_t Length, class TVector>
void FormatVector(
    TStringBuilderBase* builder,
    const char (&format)[Length],
    const TVector& vec)
{
    NYT::NDetail::TRangeFormatter<typename TVector::value_type> formatter(vec);
    NYT::NDetail::RunFormatter(builder, format, formatter);
}

template <class TVector>
void FormatVector(
    TStringBuilderBase* builder,
    TStringBuf format,
    const TVector& vec)
{
    NYT::NDetail::TRangeFormatter<typename TVector::value_type> formatter(vec);
    NYT::NDetail::RunFormatter(builder, format, formatter);
}

template <size_t Length, class TVector>
TString FormatVector(
    const char (&format)[Length],
    const TVector& vec)
{
    TStringBuilder builder;
    FormatVector(&builder, format, vec);
    return builder.Flush();
}

template <class TVector>
TString FormatVector(
    TStringBuf format,
    const TVector& vec)
{
    TStringBuilder builder;
    FormatVector(&builder, format, vec);
    return builder.Flush();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#include <util/string/cast.h>

// util/string/cast.h extension for yt and std types only
// TODO(arkady-e1ppa): Abolish ::ToString in
// favour of either NYT::ToString or
// automatic formatting wherever it is needed.
namespace NPrivate {

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

} // namespace NPrivate
