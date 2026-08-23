#pragma once

#include "public.h"

#include <library/cpp/yt/string/format.h>

#include <util/generic/strbuf.h>
#include <util/generic/typetraits.h>

#include <string>

namespace NYT::NLogging {

////////////////////////////////////////////////////////////////////////////////

//! Wraps a tag key and rejects, at compile time, the pre-migration printf spelling
//! (|WithTag("Key: %v", value)|), which would otherwise bind to |WithTag(key, value)|
//! and silently produce a tag keyed |"Key: %v"|.
class TLoggingTagKey;

////////////////////////////////////////////////////////////////////////////////

//! An opaque, pre-serialized list of logging tags.
class TLoggingTagList
{
public:
    TLoggingTagList() = default;

    //! Reconstructs a list from bytes previously produced by #GetPayload.
    explicit TLoggingTagList(TLoggingTagListPayload payload);

    template <class TValue>
    TLoggingTagList& Add(TLoggingTagKey key, const TValue& value);
    template <class... TArgs>
    TLoggingTagList& AddFormat(TLoggingTagKey key, TFormatString<TArgs...> format, TArgs&&... args);

    //! Appends every tag from #other.
    TLoggingTagList& Add(const TLoggingTagList& other);

    template <class TValue>
    [[nodiscard]] TLoggingTagList With(TLoggingTagKey key, const TValue& value) const &;
    template <class TValue>
    [[nodiscard]] TLoggingTagList With(TLoggingTagKey key, const TValue& value) &&;
    template <class... TArgs>
    [[nodiscard]] TLoggingTagList WithFormat(TLoggingTagKey key, TFormatString<TArgs...> format, TArgs&&... args) const &;
    template <class... TArgs>
    [[nodiscard]] TLoggingTagList WithFormat(TLoggingTagKey key, TFormatString<TArgs...> format, TArgs&&... args) &&;

    bool IsEmpty() const;

    //! The serialized tag section, spliced verbatim by #TTaggedPayloadWriter::AppendTags.
    const TLoggingTagListPayload& GetPayload() const;

private:
    TLoggingTagListPayload Payload_;

    template <class TValue>
    void DoAdd(TLoggingTagKey key, const TValue& value, TStringBuf spec);
};

////////////////////////////////////////////////////////////////////////////////

//! Marks a type as carrying a well-known tag under #Key.
template <class T>
struct TWellKnownLoggingTagTraits
{
    static_assert(TDependentFalse<T>, "Type does not carry a well-known logging tag; pass an explicit key");
};

////////////////////////////////////////////////////////////////////////////////

//! Views #payload without copying.
TLoggingTagListPayloadView AsView(const TLoggingTagListPayload& payload);

//! Renders the tags as |Key: Value, ...|.
void FormatValue(TStringBuilderBase* builder, TLoggingTagListPayloadView tags, TStringBuf spec);

//! Renders the tags as |Key: Value, ...|.
void FormatValue(TStringBuilderBase* builder, const TLoggingTagListPayload& tags, TStringBuf spec);

//! Renders the tags as |Key: Value, ...|.
void FormatValue(TStringBuilderBase* builder, const TLoggingTagList& tags, TStringBuf spec);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NLogging

#define TAG_INL_H_
#include "tag-inl.h"
#undef TAG_INL_H_
