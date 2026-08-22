#pragma once

#include "public.h"

#include <library/cpp/yt/string/string_builder.h>

#include <library/cpp/yt/memory/ref.h>

#include <util/generic/strbuf.h>
#include <util/generic/size_literals.h>

#include <util/system/types.h>

#include <optional>
#include <string>

namespace NYT::NLogging {

////////////////////////////////////////////////////////////////////////////////

// Plain-text events carry their message together with structured key/value tags in a
// flat, length-prefixed payload, deferring tag rendering to the consumer (the logging
// thread). These helpers own the wire format and isolate it from callers.

//! A per-thread chunk-cached string builder backing payload buffers.
/*!
 *  Generic: it knows nothing about the payload layout (see #TTaggedPayloadWriter).
 */
class TTaggedPayloadBuilder
    : public TStringBuilderBase
{
public:
    TSharedRef Flush();

    //! Appends a trivially-copyable value as raw bytes (the ui32 length prefix).
    template <class T>
    void AppendPod(const T& value);

    //! Overwrites a trivially-copyable value at #offset.
    //! #offset + sizeof(value) must not exceed the current length.
    template <class T>
    void WritePodAt(size_t offset, const T& value);

    //! For testing only.
    static void DisablePerThreadCache();

private:
    TSharedMutableRef Buffer_;

    static constexpr size_t ChunkSize = 128_KB - 64;

    void DoReset() override;
    void DoReserve(size_t newCapacity) override;
};

////////////////////////////////////////////////////////////////////////////////

//! A log message together with its tag key/value pairs, serialized into a flat,
//! opaque byte payload that #TLogEvent hands to the logging thread.
/*!
 *  Layout (all sizes are host-endian |ui32|; the transport is in-process):
 *  \code
 *    [message size][message bytes]
 *    [key size][key bytes][value size][value bytes]   x N
 *  \endcode
 *
 *  A "well-known" tag sets the high bit of its key-size field; the low 31 bits stay the
 *  key length and the key bytes are still written. Consumers may render such tags
 *  specially (see #TWellKnownLoggingTagTraits).
 *
 *  Both the producer (#TTaggedPayloadWriter) and the consumer
 *  (#TTaggedPayloadReader) own this single definition of the layout.
 */

//! The high bit of a tag's key-size field, marking a well-known tag.
constexpr ui32 WellKnownTagFlag = 1u << 31;

////////////////////////////////////////////////////////////////////////////////

//! Producer side: serializes the message and its tags into a payload.
/*!
 *  Owns a per-thread chunk-cached buffer; both the message text (#BeginMessage) and
 *  tag values (#BeginTag) are appended through the returned #TStringBuilderBase, so a
 *  formatted value is built directly into the payload without an intermediate
 *  allocation/copy. The length-prefix framing lives entirely here -- the underlying
 *  builder is format-agnostic.
 */
class TTaggedPayloadWriter
{
public:
    //! Constructs an empty writer; the message is built incrementally through the
    //! builder returned by #BeginMessage.
    TTaggedPayloadWriter() = default;

    TTaggedPayloadWriter(const TTaggedPayloadWriter&) = delete;
    TTaggedPayloadWriter& operator=(const TTaggedPayloadWriter&) = delete;

    //! Begins the message field and returns a builder for appending its text.
    //! Must be called exactly once, first.
    TStringBuilderBase* BeginMessage() &;

    //! Ends the message field, filling in its reserved length prefix. Must be
    //! called exactly once, after the message text and before any tag/#Finish.
    TTaggedPayloadWriter& EndMessage() &;

    //! Begins a tag: writes #key, then reserves the value's length prefix and returns
    //! a builder for appending the value. Lets the value be formatted directly into the
    //! payload buffer. Pair with #EndTag. Must follow #EndMessage.
    TStringBuilderBase* BeginTag(TStringBuf key) &;

    //! Like #BeginTag, but marks the tag well-known (sets #WellKnownTagFlag).
    TStringBuilderBase* BeginWellKnownTag(TStringBuf key) &;

    //! Ends the current tag, filling in the value's reserved length prefix.
    TTaggedPayloadWriter& EndTag() &;

    //! Splices an already-serialized tag section (see #TLoggingTagList::GetPayload)
    //! verbatim. Must follow #EndMessage, must not interrupt a tag, and must precede
    //! any well-known tag, which the layout requires to come last.
    TTaggedPayloadWriter& AppendTags(TLoggingTagListPayloadView tags) &;

    //! Appends a single framed keyed tag to #tags, for producers accumulating a tag
    //! section of their own (see #TLoggingTagList). The value is written by #formatter
    //! straight into #tags.
    /*!
     *  #formatter must not read or write #tags: it is being appended to -- and may be
     *  reallocated -- while #formatter runs.
     */
    template <class TFormatter>
    static void AppendTag(TLoggingTagListPayload* tags, TStringBuf key, const TFormatter& formatter);

    //! Returns the serialized payload. Must follow #EndMessage.
    TTaggedLogEventPayload Finish() &;

    //! For testing only.
    static void DisablePerThreadCache();

private:
    TTaggedPayloadBuilder Builder_;
    bool MessageStarted_ = false;
    bool MessageEnded_ = false;
    bool InTag_ = false;
    //! Offset of the length prefix currently being filled (message or tag value).
    size_t PrefixOffset_ = 0;

    //! Shared implementation of #BeginTag and #BeginWellKnownTag.
    TStringBuilderBase* DoBeginTag(TStringBuf key, bool wellKnown);
    //! Reserves a length prefix at the current position, remembering its offset.
    void ReserveLengthPrefix();
    //! Backpatches the reserved prefix with the length appended after it.
    void BackpatchLengthPrefix();
};

////////////////////////////////////////////////////////////////////////////////

//! Consumer side: parses a payload produced by #TTaggedPayloadWriter.
/*!
 *  A single-pass cursor; the returned views point into #payload, which must outlive
 *  the reader.
 *
 *  Call order: #ReadMessage must be called exactly once, first; then #TryReadTag
 *  must be called repeatedly until it returns |std::nullopt|, yielding the tags in
 *  write order.
 */
class TTaggedPayloadReader
{
public:
    struct TTag
    {
        bool IsWellKnown = false;
        TStringBuf Key;
        TStringBuf Value;
    };

    explicit TTaggedPayloadReader(const TTaggedLogEventPayload& payload);

    //! Parses a bare tag section carrying no message field (see #TLoggingTagList::GetPayload).
    //! #ReadMessage must not be called on a reader constructed this way.
    explicit TTaggedPayloadReader(TLoggingTagListPayloadView tags);

    //! Reads the message field. Must be called exactly once, before any #TryReadTag.
    TStringBuf ReadMessage();

    //! Reads the next tag; returns |std::nullopt| once the tags are exhausted.
    //! Must follow #ReadMessage.
    std::optional<TTag> TryReadTag();

private:
    const char* Current_;
    const char* const End_;
    bool MessageRead_ = false;

    //! Reads a ui32 length word.
    ui32 ReadLength();
    //! Reads #size bytes as a view.
    TStringBuf ReadBytes(ui32 size);
    //! Reads a length-prefixed string (a length word followed by its bytes).
    TStringBuf ReadString();
};

////////////////////////////////////////////////////////////////////////////////

//! Producer convenience: builds a payload carrying only #message and no tags.
TTaggedLogEventPayload MakeTaggedPayloadFromMessage(TStringBuf message);

//! Consumer convenience: returns the message, disregarding any tags. The result
//! views into #payload, which must outlive it.
TStringBuf GetMessageFromTaggedPayload(const TTaggedLogEventPayload& payload);

//! Consumer convenience: renders the payload into #builder as |Message (Key: Value, ...)|,
//! well-known tags on trailing lines. Pass a #TStringBuilder or -- to avoid allocating --
//! a #TRawFormatter.
template <class TBuilder>
void FormatTaggedPayload(TBuilder* builder, const TTaggedLogEventPayload& payload);

//! Same, returning a string.
std::string FormatTaggedPayload(const TTaggedLogEventPayload& payload);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NLogging

#define TAGGED_PAYLOAD_INL_H_
#include "tagged_payload-inl.h"
#undef TAGGED_PAYLOAD_INL_H_
