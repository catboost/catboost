#ifndef TAG_INL_H_
#error "Direct inclusion of this file is not allowed, include tag.h"
// For the sake of sane code completion.
#include "tag.h"
#endif

#include "tagged_payload.h"

#include <library/cpp/yt/string/string_builder.h>

#include <utility>

namespace NYT::NLogging {

////////////////////////////////////////////////////////////////////////////////

class TLoggingTagKey
{
public:
    template <size_t N>
    consteval TLoggingTagKey(const char (&key)[N])
        : Key_(key, N - 1)
    {
        static_assert(N >= 2, "Logging tag key must be a non-empty string literal");
        for (size_t index = 0; index + 1 < N; ++index) {
            if (key[index] == '%' || key[index] == ':') {
                // Throwing rather than calling an undefined function: a default member
                // initializer makes the compiler emit this ctor as a runtime function, and
                // an undefined sentinel would then fail at link time.
                throw "Logging tag key must not contain '%' or ':'";
            }
        }
    }

    //! Escape hatch for the few call sites that compose a key at run time.
    static TLoggingTagKey FromRuntime(TStringBuf key)
    {
        return TLoggingTagKey(key);
    }

    TStringBuf Get() const
    {
        return Key_;
    }

private:
    const TStringBuf Key_;

    explicit TLoggingTagKey(TStringBuf key)
        : Key_(key)
    { }
};

////////////////////////////////////////////////////////////////////////////////

template <class TValue>
TLoggingTagList& TLoggingTagList::Add(TLoggingTagKey key, const TValue& value)
{
    DoAdd(key, value, "v"_sb);
    return *this;
}

template <class... TArgs>
TLoggingTagList& TLoggingTagList::AddFormat(TLoggingTagKey key, TFormatString<TArgs...> format, TArgs&&... args)
{
    TTaggedPayloadWriter::AppendTag(&Payload_, key.Get(), [&] (TStringBuilderBase* builder) {
        Format(builder, format, std::forward<TArgs>(args)...);
    });
    return *this;
}

template <class TValue>
TLoggingTagList TLoggingTagList::With(TLoggingTagKey key, const TValue& value) const &
{
    auto result = *this;
    result.Add(key, value);
    return result;
}

template <class TValue>
TLoggingTagList TLoggingTagList::With(TLoggingTagKey key, const TValue& value) &&
{
    Add(key, value);
    return std::move(*this);
}

template <class... TArgs>
TLoggingTagList TLoggingTagList::WithFormat(TLoggingTagKey key, TFormatString<TArgs...> format, TArgs&&... args) const &
{
    auto result = *this;
    result.AddFormat(key, format, std::forward<TArgs>(args)...);
    return result;
}

template <class... TArgs>
TLoggingTagList TLoggingTagList::WithFormat(TLoggingTagKey key, TFormatString<TArgs...> format, TArgs&&... args) &&
{
    AddFormat(key, format, std::forward<TArgs>(args)...);
    return std::move(*this);
}

inline TLoggingTagList::TLoggingTagList(TLoggingTagListPayload payload)
    : Payload_(std::move(payload))
{ }

inline TLoggingTagList& TLoggingTagList::Add(const TLoggingTagList& other)
{
    Payload_.Underlying() += other.Payload_.Underlying();
    return *this;
}

inline bool TLoggingTagList::IsEmpty() const
{
    return Payload_.Underlying().empty();
}

inline const TLoggingTagListPayload& TLoggingTagList::GetPayload() const
{
    return Payload_;
}

template <class TValue>
void TLoggingTagList::DoAdd(TLoggingTagKey key, const TValue& value, TStringBuf spec)
{
    TTaggedPayloadWriter::AppendTag(&Payload_, key.Get(), [&] (TStringBuilderBase* builder) {
        FormatValue(builder, value, spec);
    });
}

////////////////////////////////////////////////////////////////////////////////

inline TLoggingTagListBuilder::TLoggingTagListBuilder(TLoggingTagList* tags)
    : Tags_(tags)
{ }

template <class TValue>
TLoggingTagListBuilder& TLoggingTagListBuilder::With(TLoggingTagKey key, const TValue& value)
{
    Tags_->Add(key, value);
    return *this;
}

template <class TValue>
TLoggingTagListBuilder& TLoggingTagListBuilder::WithIf(bool condition, TLoggingTagKey key, const TValue& value)
{
    return condition ? With(key, value) : *this;
}

template <class... TArgs>
TLoggingTagListBuilder& TLoggingTagListBuilder::WithFormat(
    TLoggingTagKey key,
    TFormatString<TArgs...> format,
    TArgs&&... args)
{
    Tags_->AddFormat(key, format, std::forward<TArgs>(args)...);
    return *this;
}

template <class... TArgs>
TLoggingTagListBuilder& TLoggingTagListBuilder::WithFormatIf(
    bool condition,
    TLoggingTagKey key,
    TFormatString<TArgs...> format,
    TArgs&&... args)
{
    return condition
        ? WithFormat(key, format, std::forward<TArgs>(args)...)
        : *this;
}

inline TLoggingTagListBuilder& TLoggingTagListBuilder::With(const TLoggingTagList& tags)
{
    Tags_->Add(tags);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////

inline TLoggingTagListPayloadView AsView(const TLoggingTagListPayload& payload)
{
    return TLoggingTagListPayloadView(payload.Underlying());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NLogging
