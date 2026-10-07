#pragma once

#include "public.h"

#include <library/cpp/yt/misc/lazy.h>
#include <library/cpp/yt/misc/no_op.h>

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

//! Appends keyed tags to an existing list, fluently, and invokes #functor once the chain
//! is over, i.e. when the guard dies.
/*!
 *  Lets an owner of a #TLoggingTagList offer |request->Annotate().With("Key", value)|
 *  without exposing the list itself. Holds the list by pointer and appends in place, so
 *  the chain need not be a single expression.
 *
 *  The referenced list must outlive the guard. A null #tags discards the chain without
 *  formatting anything, for owners that annotate only when someone will read the tags.
 *
 *  The functor is optional -- for owners that must act on the tags rather than merely
 *  collect them -- and is not invoked when the guard dies while an exception is
 *  propagating.
 */
template <class TFunctor = TNoOp>
class TLoggingTagListBuilderGuard
{
public:
    explicit TLoggingTagListBuilderGuard(TLoggingTagList* tags Y_LIFETIME_BOUND, TFunctor functor = {});
    ~TLoggingTagListBuilderGuard();

    TLoggingTagListBuilderGuard(const TLoggingTagListBuilderGuard&) = delete;
    TLoggingTagListBuilderGuard& operator=(const TLoggingTagListBuilderGuard&) = delete;

    template <class TValue>
    TLoggingTagListBuilderGuard& With(TLoggingTagKey key, const TValue& value);

    //! Attaches the tag only when #condition holds.
    //! NB: #value is evaluated either way unless wrapped in |YT_LAZY|.
    template <class TValue>
    TLoggingTagListBuilderGuard& WithIf(bool condition, TLoggingTagKey key, const TValue& value);

    template <class... TArgs>
    TLoggingTagListBuilderGuard& WithFormat(TLoggingTagKey key, TFormatString<TArgs...> format, TArgs&&... args);

    //! Attaches a composed tag only when #condition holds.
    //! NB: #args are evaluated either way unless wrapped in |YT_LAZY|.
    template <class... TArgs>
    TLoggingTagListBuilderGuard& WithFormatIf(
        bool condition,
        TLoggingTagKey key,
        TFormatString<TForced<TArgs>...> format,
        TArgs&&... args);

    //! Splices a pre-built list, keeping its tags individual.
    TLoggingTagListBuilderGuard& With(const TLoggingTagList& tags);

private:
    TLoggingTagList* const Tags_;
    Y_NO_UNIQUE_ADDRESS TFunctor Functor_;
    const int UncaughtExceptionCount_;
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
