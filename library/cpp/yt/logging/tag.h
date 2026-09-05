#pragma once

#include "public.h"

#include <library/cpp/yt/string/format.h>

#include <util/generic/strbuf.h>
#include <util/system/compiler.h>
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

//! Appends keyed tags to an existing list, fluently.
/*!
 *  Lets an owner of a #TLoggingTagList offer |request->Annotate().With("Key", value)|
 *  without exposing the list itself. Holds the list by pointer and appends in place, so
 *  the chain need not be a single expression and nothing is committed at the end.
 *
 *  The referenced list must outlive the builder.
 */
class TLoggingTagListBuilder
{
public:
    explicit TLoggingTagListBuilder(TLoggingTagList* tags Y_LIFETIME_BOUND);

    template <class TValue>
    TLoggingTagListBuilder& With(TLoggingTagKey key, const TValue& value);

    //! Attaches the tag only when #condition holds. NB: #value is evaluated either way.
    template <class TValue>
    TLoggingTagListBuilder& WithIf(bool condition, TLoggingTagKey key, const TValue& value);

    template <class... TArgs>
    TLoggingTagListBuilder& WithFormat(TLoggingTagKey key, TFormatString<TArgs...> format, TArgs&&... args);

    //! Attaches a composed tag only when #condition holds. NB: #args are evaluated either way.
    template <class... TArgs>
    TLoggingTagListBuilder& WithFormatIf(bool condition, TLoggingTagKey key, TFormatString<TArgs...> format, TArgs&&... args);

    //! Splices a pre-built list, keeping its tags individual.
    TLoggingTagListBuilder& With(const TLoggingTagList& tags);

private:
    TLoggingTagList* const Tags_;
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
